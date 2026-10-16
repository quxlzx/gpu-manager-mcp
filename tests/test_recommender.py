"""Ranking and candidate selection."""

from __future__ import annotations

from gpu_manager_mcp.inference.recommender import find_candidates, rank_gpus
from gpu_manager_mcp.models import GpuInfo, GpuMemory, GpuSnapshot, Manufacturer, MetricValue

GIB = 1024**3


def dev(gid: str, total_gib: float) -> GpuInfo:
    return GpuInfo(
        gpu_id=gid,
        name=gid,
        manufacturer=Manufacturer.AMD,
        backend="t",
        vram_total_bytes=int(total_gib * GIB),
    )


def snap(gid: str, free_gib: float) -> GpuSnapshot:
    return GpuSnapshot(
        gpu_id=gid,
        utilization_percent=MetricValue(value=0, unit="percent", source="t"),
        memory=GpuMemory(
            total_bytes=MetricValue(value=8 * GIB, unit="bytes", source="t"),
            used_bytes=MetricValue(value=(8 - free_gib) * GIB, unit="bytes", source="t"),
            free_bytes=MetricValue(value=free_gib * GIB, unit="bytes", source="t"),
        ),
        temperature_celsius=MetricValue(value=50, unit="celsius", source="t"),
        power_watts=MetricValue(value=100, unit="watts", source="t"),
        clock_mhz=MetricValue(value=1500, unit="MHz", source="t"),
    )


def test_rank_prefers_fitting_gpu():
    devices = [dev("small", 4), dev("big", 24)]
    snaps = {"small": snap("small", 3), "big": snap("big", 23)}
    ranking = rank_gpus(devices, snaps, 8, "q4", 8192, 1, "llama-3-8b")
    assert ranking[0]["gpu_id"] == "big"
    assert ranking[0]["estimate"]["fits"] is True


def test_find_candidates_threshold():
    devices = [dev("a", 8), dev("b", 24), dev("c", 4)]
    snaps = {"a": snap("a", 7), "b": snap("b", 20), "c": snap("c", 3)}
    out = find_candidates(devices, snaps, 16)
    assert [c["gpu_id"] for c in out] == ["b"]
    out2 = find_candidates(devices, snaps, 6)
    assert {c["gpu_id"] for c in out2} == {"a", "b"}


def test_find_candidates_falls_back_to_total_without_snapshot():
    devices = [dev("a", 8)]
    out = find_candidates(devices, {}, 6)
    assert out[0]["basis"] == "total"
