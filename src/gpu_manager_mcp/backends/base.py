"""Backend interface.

A backend discovers devices and produces normalized telemetry. Adding a new
vendor means implementing this interface and registering it in
``discovery.py`` — the MCP server itself never changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Capability, GpuInfo, GpuProcess, GpuSnapshot


class BackendError(RuntimeError):
    pass


class GPUBackend(ABC):
    #: short stable identifier used in telemetry ``source`` fields
    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        """True when the backend can talk to at least one device here."""

    @abstractmethod
    def discover(self) -> list[GpuInfo]:
        """Enumerate devices. Must not raise for missing sensors."""

    @abstractmethod
    def snapshot(self, gpu_id: str) -> GpuSnapshot:
        """One telemetry sample. Unsupported metrics must be reported as
        unavailable with a reason — never as zero."""

    def processes(self, gpu_id: str) -> list[GpuProcess]:
        """Per-process usage. Default: unsupported (empty)."""
        return []

    def supports(self, gpu_id: str, capability: Capability) -> bool:
        info = self._info(gpu_id)
        return info is not None and capability in info.capabilities

    # -- helpers -----------------------------------------------------------
    _infos: dict[str, GpuInfo]

    def __init__(self) -> None:
        self._infos = {}

    def _info(self, gpu_id: str) -> GpuInfo | None:
        return self._infos.get(gpu_id)

    def _require(self, gpu_id: str) -> GpuInfo:
        info = self._info(gpu_id)
        if info is None:
            raise BackendError(f"unknown gpu_id {gpu_id!r}; call list_gpus() for valid ids")
        return info
