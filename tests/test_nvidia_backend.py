"""NVIDIA backend with fully mocked NVML and nvidia-smi."""

from __future__ import annotations

import types

import pytest

from gpu_manager_mcp.backends import nvidia as nvidia_mod
from gpu_manager_mcp.backends.nvidia import NvidiaBackend
from gpu_manager_mcp.models import Capability


class NvmlError(Exception):
    pass


class _Mem:
    total = 24 * 1024**3
    used = 4 * 1024**3


class _Util:
    gpu = 57
    memory = 30


class _Proc:
    def __init__(self, pid, mem):
        self.pid = pid
        self.usedGpuMemory = mem


class _Pci:
    busId = b"00000000:01:00.0"


def make_fake_pynvml(power_supported: bool = True) -> types.ModuleType:
    mod = types.ModuleType("pynvml")
    mod.NVML_ERROR_NOT_SUPPORTED = "Not Supported"

    def nvmlInit():
        return None

    def nvmlDeviceGetCount():
        return 2

    def nvmlDeviceGetHandleByIndex(i):
        return i

    def nvmlDeviceGetName(h):
        return b"Tesla P40" if h == 0 else "Quadro P6000"

    def nvmlDeviceGetPciInfo(h):
        return _Pci()

    def nvmlSystemGetDriverVersion():
        return b"535.129.03"

    def nvmlDeviceGetMemoryInfo(h):
        return _Mem()

    def nvmlDeviceGetUtilizationRates(h):
        return _Util()

    def nvmlDeviceGetTemperature(h):
        return 71

    def nvmlDeviceGetPowerUsage(h):
        if not power_supported:
            raise NvmlError("NVML_ERROR_NOT_SUPPORTED: Not Supported")
        return 150_000  # mW

    def nvmlDeviceGetClockInfo(h):
        return 1480

    def nvmlDeviceGetComputeRunningProcesses(h):
        return [_Proc(111, 2 * 1024**3), _Proc(222, 512 * 1024**2)]

    for fn in (
        nvmlInit,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlDeviceGetPciInfo,
        nvmlSystemGetDriverVersion,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetTemperature,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetClockInfo,
        nvmlDeviceGetComputeRunningProcesses,
    ):
        setattr(mod, fn.__name__, fn)
    return mod


@pytest.fixture()
def nvml_backend(monkeypatch):
    monkeypatch.setattr(nvidia_mod, "pynvml", make_fake_pynvml())
    monkeypatch.setattr(nvidia_mod, "_HAVE_PYNVML", True)
    return NvidiaBackend()


def test_nvml_discovery(nvml_backend):
    assert nvml_backend.available()
    infos = nvml_backend.discover()
    assert len(infos) == 2
    assert infos[0].name == "Tesla P40"
    assert infos[1].name == "Quadro P6000"
    assert infos[0].driver_version == "535.129.03"
    assert infos[0].vram_total_bytes == 24 * 1024**3
    assert Capability.PROCESSES in infos[0].capabilities


def test_nvml_snapshot(nvml_backend):
    nvml_backend.discover()
    snap = nvml_backend.snapshot("nvidia-0000:01:00.0")
    assert snap.utilization_percent.value == 57
    assert snap.temperature_celsius.value == 71
    assert snap.power_watts.value == pytest.approx(150.0)
    assert snap.clock_mhz.value == 1480
    assert snap.memory.free_bytes.value == _Mem.total - _Mem.used


def test_nvml_unsupported_sensor_reported_not_zero(monkeypatch):
    monkeypatch.setattr(nvidia_mod, "pynvml", make_fake_pynvml(power_supported=False))
    monkeypatch.setattr(nvidia_mod, "_HAVE_PYNVML", True)
    b = NvidiaBackend()
    b.discover()
    snap = b.snapshot("nvidia-0000:01:00.0")
    assert snap.power_watts.available is False
    assert "not supported" in snap.power_watts.reason.lower()
    assert snap.power_watts.value is None


def test_nvml_processes(nvml_backend):
    nvml_backend.discover()
    procs = nvml_backend.processes("nvidia-0000:01:00.0")
    assert len(procs) == 2
    assert procs[0].used_memory_bytes.value == 2 * 1024**3


def test_no_nvml_no_smi_unavailable(monkeypatch):
    monkeypatch.setattr(nvidia_mod, "_HAVE_PYNVML", False)
    monkeypatch.setattr(nvidia_mod.shutil, "which", lambda name: None)
    b = NvidiaBackend()
    assert not b.available()
    assert b.discover() == []


def test_smi_fallback(monkeypatch):
    monkeypatch.setattr(nvidia_mod, "_HAVE_PYNVML", False)
    monkeypatch.setattr(nvidia_mod.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    class _Run:
        stdout = "Tesla P40, 24576 MiB, 1234 MiB, 12, 45, 88.5, 1392\n"

    monkeypatch.setattr(nvidia_mod.subprocess, "run", lambda *a, **k: _Run())
    b = NvidiaBackend()
    assert b.available()
    infos = b.discover()
    assert len(infos) == 1
    assert infos[0].name == "Tesla P40"
    snap = b.snapshot(infos[0].gpu_id)
    assert snap.temperature_celsius.value == 45
    assert snap.power_watts.value == 88.5


def test_smi_timeout_is_graceful(monkeypatch):
    import subprocess as sp

    monkeypatch.setattr(nvidia_mod, "_HAVE_PYNVML", False)
    monkeypatch.setattr(nvidia_mod.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise sp.TimeoutExpired(cmd="nvidia-smi", timeout=10)

    monkeypatch.setattr(nvidia_mod.subprocess, "run", boom)
    b = NvidiaBackend()
    assert b.discover() == []
