"""Vendor-agnostic GPU discovery.

Instantiates every backend, asks which are available, and merges their
device lists. Backends that raise are recorded in ``backend_errors`` instead
of taking down discovery.
"""

from __future__ import annotations

from .backends import AmdSysfsBackend, GPUBackend, IntelBackend, NvidiaBackend
from .config import Config
from .models import GpuInfo


class Discovery:
    def __init__(self, config: Config | None = None, backends: list[GPUBackend] | None = None):
        self.config = config or Config()
        self.backends = (
            backends
            if backends is not None
            else [
                NvidiaBackend(self.config),
                AmdSysfsBackend(self.config),
                IntelBackend(self.config),
            ]
        )
        self.backend_errors: dict[str, str] = {}
        self._devices: dict[str, GpuInfo] = {}
        self._backend_of: dict[str, GPUBackend] = {}

    def refresh(self) -> list[GpuInfo]:
        """Re-enumerate all devices. Devices that disappear are dropped."""
        self._devices.clear()
        self._backend_of.clear()
        self.backend_errors.clear()
        for backend in self.backends:
            try:
                if not backend.available():
                    continue
                for info in backend.discover():
                    if info.gpu_id in self._devices:
                        info.gpu_id = f"{info.gpu_id}#2"
                    self._devices[info.gpu_id] = info
                    self._backend_of[info.gpu_id] = backend
            except Exception as exc:  # a broken backend must not hide the others
                self.backend_errors[backend.name] = f"{type(exc).__name__}: {exc}"
        return list(self._devices.values())

    def devices(self) -> list[GpuInfo]:
        if not self._devices:
            self.refresh()
        return list(self._devices.values())

    def device(self, gpu_id: str) -> GpuInfo | None:
        if gpu_id not in self._devices:
            self.refresh()
        return self._devices.get(gpu_id)

    def backend_for(self, gpu_id: str) -> GPUBackend | None:
        if gpu_id not in self._backend_of:
            self.refresh()
        return self._backend_of.get(gpu_id)
