"""GPU ranking for inference workloads."""

from __future__ import annotations

from typing import Any

from ..models import GpuInfo, GpuSnapshot
from .estimator import estimate_fit


def rank_gpus(
    devices: list[GpuInfo],
    snapshots: dict[str, GpuSnapshot],
    model_size_b: float,
    quantization: str = "q4",
    context_length: int = 8192,
    parallel_requests: int = 1,
    architecture: str | None = None,
) -> list[dict]:
    """Rank GPUs by estimated fit. Prefer free VRAM, then total VRAM."""
    scored: list[dict[str, Any]] = []
    for info in devices:
        snap = snapshots.get(info.gpu_id)
        total = info.vram_total_bytes
        free = None
        if snap and snap.memory.free_bytes.available:
            free = snap.memory.free_bytes.value
        est = estimate_fit(
            model_size_b,
            quantization,
            context_length,
            parallel_requests,
            architecture,
            total,
            free,
        )
        scored.append(
            {
                "gpu_id": info.gpu_id,
                "name": info.name,
                "manufacturer": info.manufacturer.value,
                "estimate": est.model_dump(),
            }
        )

    def _rank_key(r: dict[str, Any]) -> tuple[float, float, float]:
        est = r["estimate"]
        return (
            1.0 if est["fits"] is True else 0.0,
            float(est["vram_free_gib"]) if est["vram_free_gib"] is not None else -1.0,
            float(est["vram_total_gib"]) if est["vram_total_gib"] is not None else -1.0,
        )

    scored.sort(key=_rank_key, reverse=True)
    return scored


def find_candidates(
    devices: list[GpuInfo],
    snapshots: dict[str, GpuSnapshot],
    required_vram_gb: float,
) -> list[dict]:
    """Devices whose *free* (falling back to total) memory meets the request.

    Note: GB here means GiB (1024^3), matching how GPUs report VRAM.
    """
    need = required_vram_gb * 1024**3
    out: list[dict[str, Any]] = []
    for info in devices:
        snap = snapshots.get(info.gpu_id)
        free = snap.memory.free_bytes.value if snap and snap.memory.free_bytes.available else None
        total = info.vram_total_bytes
        basis = free if free is not None else total
        if basis is None:
            continue
        if basis >= need:
            out.append(
                {
                    "gpu_id": info.gpu_id,
                    "name": info.name,
                    "free_gib": round((free or 0) / 1024**3, 2),
                    "total_gib": round(total / 1024**3, 2) if total else None,
                    "basis": "free" if free is not None else "total",
                }
            )
    out.sort(key=lambda r: float(r["free_gib"]), reverse=True)
    return out
