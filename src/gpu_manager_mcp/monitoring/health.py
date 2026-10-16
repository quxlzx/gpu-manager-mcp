"""Health and bottleneck analysis over live + historical telemetry."""

from __future__ import annotations

from ..models import GpuInfo, GpuSnapshot
from .history import HistoryStore

# Heuristic thresholds (documented, not vendor gospel).
TEMP_WARN_C = 85.0
TEMP_CRIT_C = 100.0
POWER_WARN_FRACTION = 0.95
UTIL_BUSY_PCT = 90.0
MEM_USED_FRACTION = 0.95


def health_report(info: GpuInfo, snap: GpuSnapshot, history: HistoryStore) -> dict:
    issues: list[dict] = []
    notes: list[str] = []

    temp = snap.temperature_celsius
    if temp.available and temp.value is not None:
        if temp.value >= TEMP_CRIT_C:
            issues.append(
                {
                    "severity": "critical",
                    "kind": "thermal",
                    "detail": f"temperature {temp.value:.0f}C >= {TEMP_CRIT_C:.0f}C",
                }
            )
        elif temp.value >= TEMP_WARN_C:
            issues.append(
                {
                    "severity": "warning",
                    "kind": "thermal",
                    "detail": f"temperature {temp.value:.0f}C >= {TEMP_WARN_C:.0f}C",
                }
            )
    else:
        notes.append(f"temperature unavailable: {temp.reason}")

    mem = snap.memory
    if mem.total_bytes.value and mem.used_bytes.value is not None:
        frac = mem.used_bytes.value / mem.total_bytes.value
        if frac >= MEM_USED_FRACTION:
            issues.append(
                {"severity": "warning", "kind": "memory", "detail": f"VRAM {frac * 100:.0f}% full"}
            )
    else:
        notes.append(f"memory unavailable: {mem.used_bytes.reason or mem.total_bytes.reason}")

    util = snap.utilization_percent
    if not util.available:
        notes.append(f"utilization unavailable: {util.reason}")

    return {
        "gpu_id": info.gpu_id,
        "status": "critical"
        if any(i["severity"] == "critical" for i in issues)
        else ("warning" if issues else "ok"),
        "issues": issues,
        "notes": notes,
    }


def bottleneck_report(
    info: GpuInfo, snap: GpuSnapshot, history: HistoryStore, window_seconds: float = 300.0
) -> dict:
    """Evidence-based bottleneck hints. Says so when evidence is thin."""
    findings: list[dict] = []
    insufficient: list[str] = []

    util_peak = history.peak(info.gpu_id, "utilization_percent", window_seconds)
    temp_peak = history.peak(info.gpu_id, "temperature_celsius", window_seconds)
    power_peak = history.peak(info.gpu_id, "power_watts", window_seconds)
    mem_peak = history.peak(info.gpu_id, "memory_used_bytes", window_seconds)

    if util_peak and util_peak["samples"] >= 10:
        if util_peak["avg"] >= UTIL_BUSY_PCT:
            findings.append(
                {
                    "kind": "compute",
                    "confidence": "medium",
                    "detail": f"average utilization {util_peak['avg']:.0f}% over "
                    f"{window_seconds:.0f}s ({util_peak['samples']} samples) — "
                    "GPU compute is saturated",
                }
            )
    else:
        insufficient.append("not enough utilization samples in the window")

    if temp_peak and temp_peak["max"] is not None and temp_peak["max"] >= TEMP_WARN_C:
        findings.append(
            {
                "kind": "thermal",
                "confidence": "medium",
                "detail": f"peak temperature {temp_peak['max']:.0f}C in window — "
                "possible thermal throttling",
            }
        )
    elif not temp_peak:
        insufficient.append("no temperature history in the window")

    if power_peak and snap.power_watts.available:
        cap = info.extra.get("power_limit_watts")
        if cap and power_peak["max"] and power_peak["max"] >= cap * POWER_WARN_FRACTION:
            findings.append(
                {
                    "kind": "power",
                    "confidence": "medium",
                    "detail": f"peak power {power_peak['max']:.0f}W near limit {cap:.0f}W",
                }
            )

    if (
        mem_peak
        and snap.memory.total_bytes.available
        and snap.memory.total_bytes.value
        and mem_peak["max"]
        and mem_peak["max"] / snap.memory.total_bytes.value >= MEM_USED_FRACTION
    ):
        findings.append(
            {
                "kind": "memory",
                "confidence": "medium",
                "detail": f"VRAM usage peaked at "
                f"{mem_peak['max'] / snap.memory.total_bytes.value * 100:.0f}% "
                "— memory-bound workload likely",
            }
        )
    elif not mem_peak:
        insufficient.append("no memory history in the window")

    return {
        "gpu_id": info.gpu_id,
        "window_seconds": window_seconds,
        "findings": findings,
        "insufficient_evidence": insufficient,
        "note": "Heuristics over sampled telemetry, not profiler-grade attribution.",
    }
