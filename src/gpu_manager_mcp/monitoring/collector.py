"""Snapshot collection with a short TTL cache.

Repeated tool calls within ``cache_ttl_seconds`` reuse the last snapshot so
most queries answer in well under 250 ms without re-hammering the driver.
"""

from __future__ import annotations

import time

from ..discovery import Discovery
from ..models import GpuSnapshot


class Collector:
    def __init__(self, discovery: Discovery, cache_ttl_seconds: float = 1.0):
        self.discovery = discovery
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, GpuSnapshot]] = {}

    def snapshot(self, gpu_id: str, force: bool = False) -> GpuSnapshot:
        now = time.time()
        hit = self._cache.get(gpu_id)
        if hit and not force and (now - hit[0]) < self.cache_ttl:
            return hit[1]
        backend = self.discovery.backend_for(gpu_id)
        if backend is None:
            raise KeyError(gpu_id)
        snap = backend.snapshot(gpu_id)
        self._cache[gpu_id] = (time.time(), snap)
        return snap

    def all_snapshots(self) -> dict[str, GpuSnapshot]:
        out: dict[str, GpuSnapshot] = {}
        for info in self.discovery.devices():
            try:
                out[info.gpu_id] = self.snapshot(info.gpu_id)
            except Exception:  # noqa: S112  a dead device must not sink the sweep
                continue
        return out

    def invalidate(self) -> None:
        self._cache.clear()
