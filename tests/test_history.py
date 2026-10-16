"""Bounded SQLite history."""

from __future__ import annotations

import time

from gpu_manager_mcp.models import GpuMemory, GpuSnapshot, MetricValue
from gpu_manager_mcp.monitoring.history import HistoryStore


def snap(gpu_id: str, ts: float, util: float, temp: float = 60.0) -> GpuSnapshot:
    return GpuSnapshot(
        gpu_id=gpu_id,
        timestamp=ts,
        utilization_percent=MetricValue(value=util, unit="percent", source="t"),
        memory=GpuMemory(
            total_bytes=MetricValue(value=8e9, unit="bytes", source="t"),
            used_bytes=MetricValue(value=1e9, unit="bytes", source="t"),
            free_bytes=MetricValue(value=7e9, unit="bytes", source="t"),
        ),
        temperature_celsius=MetricValue(value=temp, unit="celsius", source="t"),
        power_watts=MetricValue.unavailable("none", source="t"),
        clock_mhz=MetricValue.unavailable("none", source="t"),
    )


def test_record_and_query(tmp_path):
    h = HistoryStore(str(tmp_path / "h.sqlite3"))
    now = time.time()
    for i in range(20):
        h.record(snap("g0", now - i, util=i))
    rows = h.query("g0", "utilization_percent", 3600)
    assert len(rows) == 20
    assert rows[0]["value"] == 19 and rows[-1]["value"] == 0  # ordered oldest-first


def test_row_cap_bounds_growth(tmp_path):
    h = HistoryStore(str(tmp_path / "h.sqlite3"), max_rows=150)
    now = time.time()
    for i in range(100):
        h.record(snap("g0", now - i, util=i))  # 5 metrics per snapshot
    assert h.count() <= 150


def test_retention_window(tmp_path):
    h = HistoryStore(str(tmp_path / "h.sqlite3"), retention_hours=1)
    now = time.time()
    h.record(snap("g0", now - 7200, util=5))  # 2h old -> pruned
    h.record(snap("g0", now, util=9))
    assert h.count() == 5  # only the fresh snapshot's metrics survive


def test_peak(tmp_path):
    h = HistoryStore(str(tmp_path / "h.sqlite3"))
    now = time.time()
    for i in range(10):
        h.record(snap("g0", now - i, util=i * 5))
    p = h.peak("g0", "utilization_percent", 3600)
    assert p["max"] == 45
    assert p["min"] == 0
    assert p["samples"] == 10


def test_peak_empty_window(tmp_path):
    h = HistoryStore(str(tmp_path / "h.sqlite3"))
    assert h.peak("g0", "utilization_percent", 60) is None


def test_unavailable_metrics_not_recorded(tmp_path):
    h = HistoryStore(str(tmp_path / "h.sqlite3"))
    h.record(snap("g0", time.time(), util=1))
    assert h.query("g0", "power_watts", 3600) == []
