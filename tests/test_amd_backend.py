"""AMD sysfs backend against a fake /sys/class/drm tree."""

from __future__ import annotations

import errno

import pytest

from gpu_manager_mcp.backends.amd import AmdSysfsBackend, _read_int
from gpu_manager_mcp.models import Capability
from tests.conftest import make_amd_card


@pytest.fixture()
def backend(fake_tree):
    return AmdSysfsBackend(drm_root=fake_tree)


def test_discovery_finds_both_cards(backend):
    infos = backend.discover()
    assert len(infos) == 2
    ids = {i.gpu_id for i in infos}
    assert ids == {"amd-0000:03:00.0", "amd-0000:0f:00.0"}
    main = backend._infos["amd-0000:03:00.0"]
    assert main.name == "AMD Radeon RX 6600 XT (Navi 23 XT)"
    assert main.vram_total_bytes == 8 * 1024**3
    assert Capability.MEMORY in main.capabilities
    assert Capability.UTILIZATION in main.capabilities
    assert Capability.TEMPERATURE in main.capabilities


def test_snapshot_values_and_units(backend):
    backend.discover()
    snap = backend.snapshot("amd-0000:03:00.0")
    assert snap.utilization_percent.value == 42
    assert snap.utilization_percent.unit == "percent"
    assert snap.temperature_celsius.value == pytest.approx(55.0)
    assert snap.temperature_celsius.unit == "celsius"
    assert snap.power_watts.value == pytest.approx(75.5)
    assert snap.clock_mhz.value == pytest.approx(1620.0)
    assert snap.memory.free_bytes.value == 8 * 1024**3 - 1 * 1024**3


def test_empty_sensor_reported_unavailable_not_zero(backend):
    """The iGPU card in the fixture has an empty gpu_busy_percent."""
    backend.discover()
    snap = backend.snapshot("amd-0000:0f:00.0")
    busy = snap.utilization_percent
    assert busy.available is False
    assert busy.value is None
    assert "empty" in (busy.reason or "")


def test_power1_input_fallback(backend):
    backend.discover()
    snap = backend.snapshot("amd-0000:0f:00.0")
    assert snap.power_watts.available
    assert snap.power_watts.value == pytest.approx(12.0)


def test_unknown_gpu_id_rejected(backend):
    backend.discover()
    from gpu_manager_mcp.backends.base import BackendError

    with pytest.raises(BackendError):
        backend.snapshot("../../etc/passwd")


def test_non_amd_vendor_skipped(tmp_path):
    root = tmp_path / "drm"
    make_amd_card(root, vendor="0x8086")
    b = AmdSysfsBackend(drm_root=root)
    assert b.discover() == []
    assert not b.available()


class _RaisingPath:
    def __init__(self, err):
        self.err = err

    def read_text(self):
        raise self.err


def test_read_int_maps_ebusy():
    mv = _read_int(_RaisingPath(OSError(errno.EBUSY, "busy")))
    assert mv.available is False
    assert "busy" in mv.reason


def test_read_int_maps_eacces():
    mv = _read_int(_RaisingPath(OSError(errno.EACCES, "denied")))
    assert mv.available is False
    assert "permission" in mv.reason.lower()


def test_read_int_maps_enodev():
    mv = _read_int(_RaisingPath(OSError(errno.ENODEV, "gone")))
    assert mv.available is False


def test_device_disappears_between_discover_and_snapshot(backend, fake_tree):
    backend.discover()
    import shutil

    shutil.rmtree(fake_tree / "card0")
    snap = backend.snapshot("amd-0000:03:00.0")
    # every sensor read now fails; nothing may be silently zero
    assert snap.utilization_percent.available is False
    assert snap.memory.total_bytes.available is False
    assert snap.temperature_celsius.available is False
