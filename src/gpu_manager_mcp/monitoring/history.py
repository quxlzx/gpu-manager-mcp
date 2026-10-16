"""Bounded SQLite history store.

Rows are (gpu_id, metric, value, unit, ts). Growth is bounded two ways:
row cap (oldest deleted first) and retention window. Sampling is driven by
the caller (the MCP server samples on demand and on a background interval).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from ..models import GpuSnapshot

#: metric name -> (attribute path on GpuSnapshot, unit)
TRACKED_METRICS = {
    "utilization_percent": ("utilization_percent", "percent"),
    "memory_used_bytes": ("memory.used_bytes", "bytes"),
    "memory_total_bytes": ("memory.total_bytes", "bytes"),
    "memory_free_bytes": ("memory.free_bytes", "bytes"),
    "temperature_celsius": ("temperature_celsius", "celsius"),
    "power_watts": ("power_watts", "watts"),
    "clock_mhz": ("clock_mhz", "MHz"),
}


def _dig(snapshot: GpuSnapshot, path: str):
    obj = snapshot
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


class HistoryStore:
    def __init__(self, db_path: str, max_rows: int = 200_000, retention_hours: float = 24.0):
        self.db_path = str(db_path)
        self.max_rows = max_rows
        self.retention_hours = retention_hours
        Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS samples (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   gpu_id TEXT NOT NULL,
                   metric TEXT NOT NULL,
                   value REAL NOT NULL,
                   unit TEXT,
                   ts REAL NOT NULL
               )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_lookup ON samples (gpu_id, metric, ts)"
        )
        self._conn.commit()

    def record(self, snapshot: GpuSnapshot) -> int:
        """Insert all available metrics from one snapshot. Returns rows written."""
        rows = []
        for metric, (path, unit) in TRACKED_METRICS.items():
            mv = _dig(snapshot, path)
            if mv.available and mv.value is not None:
                rows.append((snapshot.gpu_id, metric, float(mv.value), unit, snapshot.timestamp))
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT INTO samples (gpu_id, metric, value, unit, ts) VALUES (?,?,?,?,?)", rows
            )
            self._conn.commit()
            self._prune()
        return len(rows)

    def _prune(self) -> None:
        cutoff = time.time() - self.retention_hours * 3600
        self._conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
        self._conn.execute(
            """DELETE FROM samples WHERE id NOT IN (
                   SELECT id FROM samples ORDER BY ts DESC LIMIT ?
               )""",
            (self.max_rows,),
        )
        self._conn.commit()

    def query(self, gpu_id: str, metric: str, duration_seconds: float) -> list[dict]:
        since = time.time() - duration_seconds
        with self._lock:
            cur = self._conn.execute(
                "SELECT ts, value, unit FROM samples "
                "WHERE gpu_id=? AND metric=? AND ts>=? ORDER BY ts",
                (gpu_id, metric, since),
            )
            rows = cur.fetchall()
        return [{"ts": ts, "value": v, "unit": u} for ts, v, u in rows]

    def peak(self, gpu_id: str, metric: str, duration_seconds: float) -> dict | None:
        since = time.time() - duration_seconds
        with self._lock:
            cur = self._conn.execute(
                "SELECT MAX(value), MIN(value), AVG(value), COUNT(*) FROM samples "
                "WHERE gpu_id=? AND metric=? AND ts>=?",
                (gpu_id, metric, since),
            )
            row = cur.fetchone()
        if not row or row[3] == 0:
            return None
        return {
            "metric": metric,
            "max": row[0],
            "min": row[1],
            "avg": row[2],
            "samples": row[3],
            "window_seconds": duration_seconds,
        }

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
