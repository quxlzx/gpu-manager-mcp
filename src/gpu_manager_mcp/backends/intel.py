"""Intel backend.

Primary path (Linux): sysfs for the in-kernel ``i915`` / ``xe`` drivers
(``/sys/class/drm/cardN/device`` + hwmon). ``intel_gpu_top`` is used as an
optional fallback for utilization when the JSON interface is available.

Per-process VRAM and detailed clocks on Intel GPUs are patchy across
generations; anything not exposed is reported as unavailable, honestly.
"""

from __future__ import annotations

import shutil

from ..config import Config
from ..models import (
    Capability,
    GpuInfo,
    GpuMemory,
    GpuSnapshot,
    Manufacturer,
    MetricValue,
)
from .base import GPUBackend

DRM_ROOT = "/sys/class/drm"

_INTEL_VENDORS = {"0x8086"}
_INTEL_DRIVERS = {"i915", "xe", "vgem"}


class IntelBackend(GPUBackend):
    name = "intel-sysfs"

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config or Config()
        self._gpu_top = shutil.which("intel_gpu_top")

    def available(self) -> bool:
        return self._find_cards() != [] or self._gpu_top is not None

    def _find_cards(self) -> list:
        import glob
        import os

        cards = []
        for dev_path in sorted(glob.glob(f"{DRM_ROOT}/card[0-9]*/device")):
            try:
                with open(os.path.join(dev_path, "vendor")) as fh:
                    vendor = fh.read().strip().lower()
                driver = os.path.basename(os.path.realpath(os.path.join(dev_path, "driver")))
            except OSError:
                continue
            if vendor in _INTEL_VENDORS and driver in _INTEL_DRIVERS:
                cards.append(dev_path)
        return cards

    def discover(self) -> list[GpuInfo]:
        infos: list[GpuInfo] = []
        for dev_path in self._find_cards():
            import os

            gpu_id = f"intel-{os.path.basename(os.path.dirname(dev_path))}"
            caps: list[Capability] = []
            if os.path.exists(os.path.join(dev_path, "gpu_busy_percent")):
                caps.append(Capability.UTILIZATION)
            if glob_hwmon(dev_path):
                caps += [Capability.TEMPERATURE, Capability.POWER, Capability.CLOCKS]
            info = GpuInfo(
                gpu_id=gpu_id,
                name=f"Intel GPU ({os.path.basename(os.path.dirname(dev_path))})",
                manufacturer=Manufacturer.INTEL,
                backend=self.name,
                capabilities=sorted(set(caps), key=lambda c: c.value),
                extra={"sysfs_path": dev_path},
            )
            self._infos[gpu_id] = info
            infos.append(info)
        return infos

    def snapshot(self, gpu_id: str) -> GpuSnapshot:
        import os

        info = self._require(gpu_id)
        dev = info.extra["sysfs_path"]

        def read_file(path, scale=1.0, unit=None):
            try:
                with open(path) as fh:
                    return MetricValue(
                        value=int(fh.read().strip()) * scale, unit=unit, source="intel-sysfs"
                    )
            except (OSError, ValueError):
                return MetricValue.unavailable(
                    "sensor unavailable on this Intel GPU/driver", source="intel-sysfs"
                )

        busy = read_file(os.path.join(dev, "gpu_busy_percent"), unit="percent")
        na = MetricValue.unavailable(
            "not exposed by the i915/xe sysfs interface", source="intel-sysfs"
        )
        hw = glob_hwmon(dev)
        temp = power = clock = na
        if hw:
            temp = read_file(os.path.join(hw, "temp1_input"), 1 / 1000.0, "celsius")
            power = read_file(os.path.join(hw, "power1_input"), 1 / 1_000_000.0, "watts")
            clock = read_file(os.path.join(hw, "freq1_input"), 1 / 1_000_000.0, "MHz")
        mem = GpuMemory(total_bytes=na, used_bytes=na, free_bytes=na)
        return GpuSnapshot(
            gpu_id=gpu_id,
            utilization_percent=busy,
            memory=mem,
            temperature_celsius=temp,
            power_watts=power,
            clock_mhz=clock,
        )


def glob_hwmon(dev_path: str) -> str | None:
    import glob
    import os

    for hw in sorted(glob.glob(os.path.join(dev_path, "hwmon", "hwmon*"))):
        try:
            with open(os.path.join(hw, "name")) as fh:
                if fh.read().strip() in ("i915", "xe", "i915_gpu", "xe_gpu"):
                    return hw
        except OSError:
            continue
    return None
