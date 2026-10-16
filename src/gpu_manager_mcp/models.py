"""Normalized data models shared by all GPU backends."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Manufacturer(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


class Capability(StrEnum):
    UTILIZATION = "utilization"
    MEMORY = "memory"
    TEMPERATURE = "temperature"
    POWER = "power"
    CLOCKS = "clocks"
    PROCESSES = "processes"


class MetricValue(BaseModel):
    """A single telemetry reading.

    ``available`` is False when the sensor is missing, unsupported or the
    read failed; ``reason`` then explains why. Values are never silently
    replaced with zero.
    """

    available: bool = True
    value: float | None = None
    unit: str | None = None
    source: str = ""
    timestamp: float = Field(default_factory=time.time)
    reason: str | None = None

    @classmethod
    def unavailable(cls, reason: str, source: str = "") -> MetricValue:
        return cls(available=False, reason=reason, source=source)


class GpuInfo(BaseModel):
    gpu_id: str
    name: str
    manufacturer: Manufacturer
    backend: str
    pci_bus_id: str | None = None
    driver_version: str | None = None
    vram_total_bytes: int | None = None
    capabilities: list[Capability] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class GpuMemory(BaseModel):
    total_bytes: MetricValue
    used_bytes: MetricValue
    free_bytes: MetricValue


class GpuProcess(BaseModel):
    pid: int | None = None
    name: str | None = None
    used_memory_bytes: MetricValue
    engine: str | None = None


class GpuSnapshot(BaseModel):
    """One full telemetry sample for one GPU."""

    gpu_id: str
    timestamp: float = Field(default_factory=time.time)
    utilization_percent: MetricValue
    memory: GpuMemory
    temperature_celsius: MetricValue
    power_watts: MetricValue
    clock_mhz: MetricValue
    fan_percent: MetricValue | None = None
