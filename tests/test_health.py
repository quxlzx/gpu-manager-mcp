"""Health + bottleneck heuristics, including insufficient-evidence paths."""

from __future__ import annotations

import time

from gpu_manager_mcp.models import GpuInfo, GpuMemory, GpuSnapshot, Manufacturer, MetricValue
from gpu_manager_mcp.monitoring.health import bottleneck_report, health_report
from gpu_manager_mcp.monitoring.history import HistoryStore

GIB = 1024**3


def info(gid="g0") -> GpuInfo:
    return GpuInfo(
        gpu_id=gid, name="t", manufacturer=Manufacturer.AMD, backend="t", vram_total_bytes=8 * GIB
    )


def snap(gid="g0", temp=60.0, util=50.0, used_frac=0.5, mem_available=True) -> GpuSnapshot:
    total = MetricValue(value=8 * GIB, unit="bytes", source="t")
    used = (
        MetricValue(value=used_frac * 8 * GIB, unit="bytes", source="t")
        if mem_available
        else MetricValue.unavailable("sensor busy", source="t")
    )
    return GpuSnapshot(
        gpu_id=gid,
        utilization_percent=MetricValue(value=util, unit="percent", source="t"),
        memory=GpuMemory(
            total_bytes=total,
            used_bytes=used,
            free_bytes=MetricValue(value=(1 - used_frac) * 8 * GIB, unit="bytes", source="t"),
        ),
        temperature_celsius=MetricValue(value=temp, unit="celsius", source="t")
        if mem_available
        else MetricValue.unavailable("sensor busy", source="t"),
        power_watts=MetricValue(value=100, unit="watts", source="t"),
        clock_mhz=MetricValue(value=1500, unit="MHz", source="t"),
    )


def store(tmp_path) -> HistoryStore:
    return HistoryStore(str(tmp_path / "h.sqlite3"))


def test_healthy(tmp_path):
    r = health_report(info(), snap(), store(tmp_path))
    assert r["status"] == "ok"


def test_thermal_warning(tmp_path):
    r = health_report(info(), snap(temp=90), store(tmp_path))
    assert r["status"] == "warning"
    assert any(i["kind"] == "thermal" for i in r["issues"])


def test_thermal_critical(tmp_path):
    r = health_report(info(), snap(temp=105), store(tmp_path))
    assert r["status"] == "critical"


def test_memory_full_warning(tmp_path):
    r = health_report(info(), snap(used_frac=0.98), store(tmp_path))
    assert any(i["kind"] == "memory" for i in r["issues"])


def test_unavailable_metrics_become_notes(tmp_path):
    r = health_report(info(), snap(mem_available=False), store(tmp_path))
    assert any("temperature unavailable" in n for n in r["notes"])
    assert any("memory unavailable" in n for n in r["notes"])


def test_bottleneck_compute_saturated(tmp_path):
    h = store(tmp_path)
    now = time.time()
    for i in range(30):
        h.record(snap(util=97, temp=60).model_copy(update={"timestamp": now - i}))
    rep = bottleneck_report(info(), snap(util=97), h, 600)
    assert any(f["kind"] == "compute" for f in rep["findings"])


def test_bottleneck_insufficient_evidence(tmp_path):
    rep = bottleneck_report(info(), snap(), store(tmp_path), 60)
    assert rep["findings"] == []
    assert rep["insufficient_evidence"]


def test_bottleneck_memory_bound(tmp_path):
    h = store(tmp_path)
    now = time.time()
    for i in range(30):
        h.record(snap(used_frac=0.99, util=40).model_copy(update={"timestamp": now - i}))
    rep = bottleneck_report(info(), snap(used_frac=0.99), h, 600)
    assert any(f["kind"] == "memory" for f in rep["findings"])
