"""AMD backend.

Primary path (Linux): the amdgpu driver's sysfs interface
(``/sys/class/drm/cardN/device`` + its hwmon child). This works on any Linux
with the in-kernel amdgpu driver — no ROCm required.

Fallback path: the ``rocm-smi`` CLI when present (adds per-process info and
fills gaps). Windows support is not implemented in this release; the backend
reports itself unavailable there instead of guessing.
"""

from __future__ import annotations

import errno
import shutil
import subprocess
from pathlib import Path

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

DRM_ROOT = Path("/sys/class/drm")

#: Small human-readable map for common AMD PCI device ids. Unknown ids fall
#: back to the raw hex so the device is still listed honestly.
DEVICE_NAMES = {
    0x73FF: "AMD Radeon RX 6600 XT (Navi 23 XT)",
    0x73BF: "AMD Radeon RX 6600 (Navi 23)",
    0x1638: "AMD Radeon Graphics (Cezanne iGPU)",
    0x164C: "AMD Radeon Graphics (Renoir iGPU)",
    0x15D8: "AMD Radeon Graphics (Vega iGPU)",
    0x744C: "AMD Radeon RX 7900 XT (Navi 31)",
    0x744E: "AMD Radeon RX 7900 XTX (Navi 31)",
    0x7400: "AMD Radeon RX 7600 (Navi 33)",
}

_OSMAP = {
    errno.EACCES: "permission denied reading sensor (try running as root or a udev rule)",
    errno.EBUSY: "sensor busy — the GPU power management interface refused the read "
    "(common when the card is in a low-power state)",
    errno.ENODEV: "sensor not present on this card",
    errno.EOPNOTSUPP: "sensor unsupported by this driver/card",
    errno.ENOENT: "sensor file missing",
    errno.EIO: "hardware returned an I/O error for this sensor",
}


def _read_int(path: Path) -> MetricValue:
    """Read one sysfs integer file, mapping failures to honest reasons."""
    try:
        raw = path.read_text().strip()
    except FileNotFoundError:
        return MetricValue.unavailable("sensor file missing", source="amd-sysfs")
    except OSError as exc:
        reason = _OSMAP.get(exc.errno or 0, f"read failed: {exc.strerror or exc}")
        return MetricValue.unavailable(reason, source="amd-sysfs")
    if raw == "":
        return MetricValue.unavailable(
            "sensor returned empty value (driver refused the read)", source="amd-sysfs"
        )
    try:
        return MetricValue(value=int(raw), source="amd-sysfs")
    except ValueError:
        return MetricValue.unavailable(
            f"sensor returned non-numeric value {raw[:32]!r}", source="amd-sysfs"
        )


def _as_int(mv: MetricValue) -> int | None:
    return int(mv.value) if mv.value is not None else None


class AmdSysfsBackend(GPUBackend):
    name = "amd-sysfs"

    def __init__(self, config: Config | None = None, drm_root: Path | None = None) -> None:
        super().__init__()
        self.config = config or Config()
        self.drm_root = Path(drm_root) if drm_root is not None else DRM_ROOT
        self._rocm_smi = shutil.which("rocm-smi")

    def available(self) -> bool:
        return self._find_cards() != []

    # -- discovery ---------------------------------------------------------
    def _find_cards(self) -> list[Path]:
        if not self.drm_root.is_dir():
            return []
        cards: list[Path] = []
        for card in sorted(self.drm_root.glob("card[0-9]*")):
            dev = card / "device"
            try:
                vendor = (dev / "vendor").read_text().strip()
            except OSError:
                continue
            if vendor.lower() == "0x1002":
                cards.append(dev)
        return cards

    def _hwmon_dir(self, dev: Path) -> Path | None:
        for hw in sorted((dev / "hwmon").glob("hwmon*")) if (dev / "hwmon").is_dir() else []:
            try:
                if hw.joinpath("name").read_text().strip() == "amdgpu":
                    return hw
            except OSError:
                continue
        return None

    def discover(self) -> list[GpuInfo]:
        infos: list[GpuInfo] = []
        for dev in self._find_cards():
            try:
                device_id = int((dev / "device").read_text().strip(), 16)
            except OSError:
                device_id = 0
            pci = None
            try:
                uevent = (dev / "uevent").read_text()
                for line in uevent.splitlines():
                    if line.startswith("PCI_SLOT_NAME="):
                        pci = line.split("=", 1)[1].strip()
            except OSError:
                pass
            gpu_id = f"amd-{pci or dev.name}"
            vram = _read_int(dev / "mem_info_vram_total")
            if not vram.available:
                vram = _read_int(dev / "mem_info_vis_vram_total")
            busy = _read_int(dev / "gpu_busy_percent")
            hw = self._hwmon_dir(dev)
            caps = [Capability.MEMORY] if vram.available else []
            if busy.available:
                caps.append(Capability.UTILIZATION)
            if hw is not None:
                if (hw / "temp1_input").exists():
                    caps.append(Capability.TEMPERATURE)
                if (hw / "power1_average").exists() or (hw / "power1_input").exists():
                    caps.append(Capability.POWER)
                if (hw / "freq1_input").exists():
                    caps.append(Capability.CLOCKS)
            if self._rocm_smi:
                caps.append(Capability.PROCESSES)
            driver = None
            try:
                driver = Path("/sys/module/amdgpu/version").read_text().strip() or None
            except OSError:
                pass
            info = GpuInfo(
                gpu_id=gpu_id,
                name=DEVICE_NAMES.get(device_id, f"AMD GPU (device id 0x{device_id:04x})"),
                manufacturer=Manufacturer.AMD,
                backend=self.name,
                pci_bus_id=pci,
                driver_version=driver,
                vram_total_bytes=_as_int(vram),
                capabilities=sorted(set(caps), key=lambda c: c.value),
                extra={"sysfs_path": str(dev), "device_id": hex(device_id)},
            )
            self._infos[gpu_id] = info
            infos.append(info)
        return infos

    # -- telemetry ---------------------------------------------------------
    def snapshot(self, gpu_id: str) -> GpuSnapshot:
        info = self._require(gpu_id)
        dev = Path(info.extra["sysfs_path"])
        hw = self._hwmon_dir(dev)

        busy = _read_int(dev / "gpu_busy_percent")
        if busy.available:
            busy.unit = "percent"

        total = _read_int(dev / "mem_info_vram_total")
        if not total.available:
            total = _read_int(dev / "mem_info_vis_vram_total")
        used = _read_int(dev / "mem_info_vram_used")
        if not used.available:
            used = _read_int(dev / "mem_info_vis_vram_used")
        for m in (total, used):
            m.unit = "bytes"
        if total.value is not None and used.value is not None:
            free = MetricValue(
                value=max(int(total.value - used.value), 0),
                unit="bytes",
                source="amd-sysfs(derived)",
            )
        else:
            free = MetricValue.unavailable(
                "requires both total and used VRAM sensors", source="amd-sysfs"
            )

        temp = power = clock = MetricValue.unavailable(
            "no amdgpu hwmon directory", source="amd-sysfs"
        )
        fan = None
        if hw is not None:
            temp = _read_int(hw / "temp1_input")
            if temp.value is not None:
                temp.value /= 1000.0
                temp.unit = "celsius"
            power = _read_int(hw / "power1_average")
            if not power.available:
                power = _read_int(hw / "power1_input")
            if power.value is not None:
                power.value /= 1_000_000.0
                power.unit = "watts"
            clock = _read_int(hw / "freq1_input")
            if clock.value is not None:
                clock.value /= 1_000_000.0
                clock.unit = "MHz"
            fanv = _read_int(hw / "fan1_input")
            if fanv.available:
                fanv.unit = "rpm"
                fan = fanv

        return GpuSnapshot(
            gpu_id=gpu_id,
            utilization_percent=busy,
            memory=GpuMemory(total_bytes=total, used_bytes=used, free_bytes=free),
            temperature_celsius=temp,
            power_watts=power,
            clock_mhz=clock,
            fan_percent=fan,
        )

    # -- processes (rocm-smi fallback) --------------------------------------
    def processes(self, gpu_id: str) -> list[GpuProcess]:
        info = self._require(gpu_id)
        if not self._rocm_smi:
            return []
        pci = (info.pci_bus_id or "").lower()
        # Fixed argv; gpu_id is only ever compared, never interpolated.
        try:
            out = subprocess.run(  # noqa: S603  fixed argv from shutil.which, no user input
                [self._rocm_smi, "--showpids", "--format=csv"],
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout_seconds,
            )
        except (subprocess.TimeoutExpired, OSError):
            return []
        procs: list[GpuProcess] = []
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5 or parts[0] == "pid":
                continue
            pid_s, name, _, mem_kb, loc = parts[0], parts[1], parts[2], parts[3], parts[4]
            if pci and pci not in loc.lower():
                continue
            try:
                pid = int(pid_s)
                mem = int(mem_kb) * 1024
            except ValueError:
                continue
            procs.append(
                GpuProcess(
                    pid=pid,
                    name=name or None,
                    used_memory_bytes=MetricValue(value=mem, unit="bytes", source="rocm-smi"),
                )
            )
        return procs
