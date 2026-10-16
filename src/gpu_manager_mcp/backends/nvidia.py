"""NVIDIA backend.

Primary path: NVML via the ``pynvml`` bindings (``nvidia-ml-py`` package,
optional dependency). Fallback: parsing ``nvidia-smi --query`` CSV output
when NVML bindings are missing but the driver CLI is present.

Every NVML call is wrapped: unsupported sensors on GeForce/Tesla/Quadro
combinations are reported as unavailable with the driver's own error text.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess

from ..config import Config
from ..models import (
    Capability,
    GpuInfo,
    GpuMemory,
    GpuProcess,
    GpuSnapshot,
    Manufacturer,
    MetricValue,
)
from .base import GPUBackend

try:  # optional dependency: nvidia-ml-py
    import pynvml  # type: ignore

    _HAVE_PYNVML = True
except ImportError:  # pragma: no cover - depends on host
    pynvml = None
    _HAVE_PYNVML = False


def _nvml_unavailable(exc: Exception) -> MetricValue:
    text = str(exc)
    if "Not Supported" in text or "NOT_SUPPORTED" in text:
        reason = "sensor not supported by this GPU/driver"
    elif "Insufficient Permissions" in text or "PERMISSIONS" in text:
        reason = "insufficient permissions for NVML (CAP_SYS_ADMIN or root may be required)"
    else:
        reason = f"NVML error: {text[:120]}"
    return MetricValue.unavailable(reason, source="nvml")


def _csv_num(cell: str, scale: float = 1.0) -> int | None:
    try:
        return int(float(cell.strip().split()[0]) * scale)
    except (ValueError, IndexError):
        return None


class NvidiaBackend(GPUBackend):
    name = "nvidia"

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config or Config()
        self._nvml_ok = False
        self._smi_path = shutil.which("nvidia-smi")
        if _HAVE_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_ok = True
            except Exception:
                self._nvml_ok = False

    def available(self) -> bool:
        if self._nvml_ok:
            try:
                return pynvml.nvmlDeviceGetCount() > 0
            except Exception:
                return False
        return self._smi_path is not None

    # -- discovery ---------------------------------------------------------
    def discover(self) -> list[GpuInfo]:
        infos: list[GpuInfo] = []
        if self._nvml_ok:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                pci = None
                try:
                    raw = pynvml.nvmlDeviceGetPciInfo(handle).busId
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    # NVML bus ids look like "0000:01:00.0" (domain:bus:dev.func)
                    tail = raw.split(":", 1)[1] if ":" in raw else raw
                    pci = f"0000:{tail}"
                except Exception:  # noqa: S110  optional field; None is honest
                    pass
                try:
                    driver = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver, bytes):
                        driver = driver.decode()
                except Exception:
                    driver = None
                mem_total = None
                try:
                    mem_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
                except Exception:  # noqa: S110  optional field; None is honest
                    pass
                caps = [Capability.MEMORY, Capability.UTILIZATION, Capability.TEMPERATURE]
                for cap, fn in (
                    (Capability.POWER, pynvml.nvmlDeviceGetPowerUsage),
                    (Capability.CLOCKS, pynvml.nvmlDeviceGetClockInfo),
                    (Capability.PROCESSES, pynvml.nvmlDeviceGetComputeRunningProcesses),
                ):
                    try:
                        fn(handle)
                        caps.append(cap)
                    except Exception:  # noqa: S110  capability probe; absence is expected
                        pass
                gpu_id = f"nvidia-{pci or i}"
                info = GpuInfo(
                    gpu_id=gpu_id,
                    name=name,
                    manufacturer=Manufacturer.NVIDIA,
                    backend=self.name,
                    pci_bus_id=pci,
                    driver_version=driver,
                    vram_total_bytes=mem_total,
                    capabilities=sorted(set(caps), key=lambda c: c.value),
                    extra={"nvml_index": i},
                )
                self._infos[gpu_id] = info
                infos.append(info)
            return infos

        # nvidia-smi CSV fallback
        if not self._smi_path:
            return infos
        query = "name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm"
        try:
            out = subprocess.run(  # noqa: S603  fixed argv from shutil.which; query is a constant
                [self._smi_path, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout_seconds,
            )
        except (subprocess.TimeoutExpired, OSError):
            return infos
        for idx, row in enumerate(csv.reader(io.StringIO(out.stdout))):
            if len(row) < 7:
                continue
            gpu_id = f"nvidia-smi-{idx}"
            info = GpuInfo(
                gpu_id=gpu_id,
                name=row[0].strip(),
                manufacturer=Manufacturer.NVIDIA,
                backend="nvidia-smi",
                vram_total_bytes=_csv_num(row[1], 1024 * 1024),
                capabilities=[
                    Capability.MEMORY,
                    Capability.UTILIZATION,
                    Capability.TEMPERATURE,
                    Capability.POWER,
                    Capability.CLOCKS,
                ],
                extra={"smi_row": row},
            )
            self._infos[gpu_id] = info
            infos.append(info)
        return infos

    # -- telemetry ---------------------------------------------------------
    def snapshot(self, gpu_id: str) -> GpuSnapshot:
        info = self._require(gpu_id)
        if info.backend == "nvidia-smi":
            return self._snapshot_smi(info)
        handle = pynvml.nvmlDeviceGetHandleByIndex(info.extra["nvml_index"])

        def probe(fn, scale=1.0, unit=None):
            try:
                v = fn(handle)
                if scale != 1.0:
                    v = v * scale
                return MetricValue(value=v, unit=unit, source="nvml")
            except Exception as exc:
                return _nvml_unavailable(exc)

        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util = MetricValue(value=getattr(u, "gpu", u), unit="percent", source="nvml")
        except Exception as exc:
            util = _nvml_unavailable(exc)
        total = used = MetricValue.unavailable("memory read failed", source="nvml")
        free = MetricValue.unavailable("requires total and used memory", source="nvml")
        try:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = MetricValue(value=m.total, unit="bytes", source="nvml")
            used = MetricValue(value=m.used, unit="bytes", source="nvml")
            free = MetricValue(value=m.total - m.used, unit="bytes", source="nvml(derived)")
        except Exception as exc:
            total = used = _nvml_unavailable(exc)
        temp = probe(pynvml.nvmlDeviceGetTemperature, unit="celsius")
        power = probe(pynvml.nvmlDeviceGetPowerUsage, scale=1 / 1000.0, unit="watts")
        clock = probe(pynvml.nvmlDeviceGetClockInfo, unit="MHz")
        return GpuSnapshot(
            gpu_id=gpu_id,
            utilization_percent=util,
            memory=GpuMemory(total_bytes=total, used_bytes=used, free_bytes=free),
            temperature_celsius=temp,
            power_watts=power,
            clock_mhz=clock,
        )

    def _snapshot_smi(self, info: GpuInfo) -> GpuSnapshot:
        row = info.extra["smi_row"]

        def num(col: int, scale=1.0, unit=None) -> MetricValue:
            try:
                return MetricValue(value=float(row[col]) * scale, unit=unit, source="nvidia-smi")
            except (ValueError, IndexError):
                return MetricValue.unavailable("nvidia-smi returned no value", source="nvidia-smi")

        total = num(1, 1024 * 1024, "bytes")
        used = num(2, 1024 * 1024, "bytes")
        free = (
            MetricValue(value=total.value - used.value, unit="bytes", source="nvidia-smi(derived)")
            if total.value is not None and used.value is not None
            else MetricValue.unavailable("missing memory sensors", source="nvidia-smi")
        )
        return GpuSnapshot(
            gpu_id=info.gpu_id,
            utilization_percent=num(3, unit="percent"),
            memory=GpuMemory(total_bytes=total, used_bytes=used, free_bytes=free),
            temperature_celsius=num(4, unit="celsius"),
            power_watts=num(5, unit="watts"),
            clock_mhz=num(6, unit="MHz"),
        )

    def processes(self, gpu_id: str) -> list[GpuProcess]:
        info = self._require(gpu_id)
        if not self._nvml_ok:
            return []
        handle = pynvml.nvmlDeviceGetHandleByIndex(info.extra["nvml_index"])
        try:
            entries = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception:
            return []
        procs = []
        for e in entries:
            mem = getattr(e, "usedGpuMemory", None)
            procs.append(
                GpuProcess(
                    pid=getattr(e, "pid", None),
                    name=None,
                    used_memory_bytes=MetricValue(value=mem, unit="bytes", source="nvml")
                    if mem is not None
                    else MetricValue.unavailable(
                        "driver does not report per-process memory", source="nvml"
                    ),
                )
            )
        return procs
