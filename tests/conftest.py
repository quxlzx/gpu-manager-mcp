"""Shared fixtures: a fake /sys/class/drm tree and fake backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_manager_mcp.backends.base import GPUBackend
from gpu_manager_mcp.models import (
    Capability,
    GpuInfo,
    GpuMemory,
    GpuSnapshot,
    Manufacturer,
    MetricValue,
)


def write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def make_amd_card(
    root: Path,
    card: str = "card9",
    vendor: str = "0x1002",
    device: str = "0x73ff",
    pci: str = "0000:99:00.0",
    busy: str = "42",
    vram_total: str = "8589934592",
    vram_used: str = "1073741824",
    hwmon: dict[str, str] | None = None,
) -> Path:
    """Build one plausible amdgpu sysfs device directory."""
    dev = root / card / "device"
    write(dev / "vendor", vendor + "\n")
    write(dev / "device", device + "\n")
    write(dev / "uevent", f"DRIVER=amdgpu\nMAJOR=226\nMINOR=0\nPCI_SLOT_NAME={pci}\n")
    write(dev / "gpu_busy_percent", busy + "\n")
    write(dev / "mem_info_vram_total", vram_total + "\n")
    write(dev / "mem_info_vram_used", vram_used + "\n")
    hw = (
        hwmon
        if hwmon is not None
        else {
            "name": "amdgpu",
            "temp1_input": "55000",
            "power1_average": "75500000",
            "freq1_input": "1620000000",
        }
    )
    hwmon_dir = dev / "hwmon" / "hwmon3"
    for name, value in hw.items():
        write(hwmon_dir / name, value + "\n")
    return dev


class FakeBackend(GPUBackend):
    """Deterministic backend for discovery/server tests."""

    name = "fake"

    def __init__(self, gpu_ids=("fake-0",), fail_discover: bool = False) -> None:
        super().__init__()
        self.gpu_ids = list(gpu_ids)
        self.fail_discover = fail_discover

    def available(self) -> bool:
        return True

    def discover(self) -> list[GpuInfo]:
        if self.fail_discover:
            raise RuntimeError("simulated backend crash")
        infos = []
        for gid in self.gpu_ids:
            info = GpuInfo(
                gpu_id=gid,
                name="Fake GPU 8GB",
                manufacturer=Manufacturer.UNKNOWN,
                backend=self.name,
                vram_total_bytes=8 * 1024**3,
                capabilities=[Capability.MEMORY, Capability.UTILIZATION, Capability.TEMPERATURE],
            )
            self._infos[gid] = info
            infos.append(info)
        return infos

    def snapshot(self, gpu_id: str) -> GpuSnapshot:
        self._require(gpu_id)
        total = MetricValue(value=8 * 1024**3, unit="bytes", source="fake")
        used = MetricValue(value=1 * 1024**3, unit="bytes", source="fake")
        return GpuSnapshot(
            gpu_id=gpu_id,
            utilization_percent=MetricValue(value=33.0, unit="percent", source="fake"),
            memory=GpuMemory(
                total_bytes=total,
                used_bytes=used,
                free_bytes=MetricValue(value=7 * 1024**3, unit="bytes", source="fake"),
            ),
            temperature_celsius=MetricValue(value=60.0, unit="celsius", source="fake"),
            power_watts=MetricValue.unavailable("no sensor", source="fake"),
            clock_mhz=MetricValue.unavailable("no sensor", source="fake"),
        )


@pytest.fixture()
def fake_tree(tmp_path: Path) -> Path:
    root = tmp_path / "drm"
    make_amd_card(root, card="card0", pci="0000:03:00.0")
    make_amd_card(
        root,
        card="card1",
        device="0x1638",
        pci="0000:0f:00.0",
        busy="",
        vram_total="536870912",
        vram_used="16000000",
        hwmon={
            "name": "amdgpu",
            "temp1_input": "48000",
            "power1_input": "12000000",
            "freq1_input": "400000000",
        },
    )
    return root
