"""LLM memory-fit estimation.

These are order-of-magnitude engineering estimates, not measurements. The
KV-cache term depends on architecture (layers, KV heads, head dim); when the
architecture is unknown we fall back to documented default assumptions and
say so in the result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

GIB = 1024**3

#: bytes per parameter by weight quantization
BYTES_PER_PARAM = {
    "f16": 2.0,
    "f8": 1.0,
    "q8": 1.0,
    "q6": 0.75,
    "q5": 0.625,
    "q4": 0.5,
    "q4_0": 0.5,
    "q4_k_m": 0.5625,
    "q3": 0.375,
    "q2": 0.25,
}

#: (n_layers, n_kv_heads, head_dim, kv_dtype_bytes) for common architectures.
#: Approximate public specs; extend via the architecture parameter.
ARCH_KV = {
    "llama-3-8b": (32, 8, 128, 2),
    "llama-3.1-8b": (32, 8, 128, 2),
    "llama-3-70b": (80, 8, 128, 2),
    "llama-2-7b": (32, 32, 128, 2),
    "llama-2-13b": (40, 40, 128, 2),
    "mistral-7b": (32, 8, 128, 2),
    "mixtral-8x7b": (32, 8, 128, 2),
    "phi-3-mini": (32, 32, 96, 2),
    "gemma-2-9b": (42, 16, 256, 2),
    "qwen2-7b": (28, 4, 128, 2),
    "deepseek-v2-lite": (28, 1, 192, 2),
}

#: used when architecture metadata is unavailable — a GQA-era default
DEFAULT_KV = (32, 8, 128, 2)


class FitEstimate(BaseModel):
    available: bool
    weights_gib: float
    runtime_overhead_gib: float
    kv_cache_gib_per_request: float
    total_gib: float
    vram_total_gib: float | None
    vram_free_gib: float | None
    fits: bool | None
    headroom_gib: float | None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def kv_bytes_per_token(architecture: str | None) -> tuple[float, bool]:
    arch = (architecture or "").lower()
    spec, defaulted = ARCH_KV.get(arch, DEFAULT_KV), arch not in ARCH_KV
    n_layers, n_kv_heads, head_dim, kv_dtype = spec
    return 2.0 * n_layers * n_kv_heads * head_dim * kv_dtype, defaulted


def estimate_fit(
    model_size_b: float,
    quantization: str = "q4",
    context_length: int = 8192,
    parallel_requests: int = 1,
    architecture: str | None = None,
    vram_total_bytes: float | None = None,
    vram_free_bytes: float | None = None,
) -> FitEstimate:
    """Estimate VRAM needs for an LLM workload.

    ``model_size_b`` is the parameter count in billions (active params for
    MoE models). Returns a structured estimate with explicit assumptions.
    """
    warnings: list[str] = []
    assumptions: list[str] = []

    if model_size_b <= 0:
        return FitEstimate(
            available=False,
            weights_gib=0,
            runtime_overhead_gib=0,
            kv_cache_gib_per_request=0,
            total_gib=0,
            vram_total_gib=None,
            vram_free_gib=None,
            fits=None,
            headroom_gib=None,
            warnings=["model_size_b must be > 0"],
        )
    quant = (quantization or "q4").lower()
    if quant not in BYTES_PER_PARAM:
        warnings.append(
            f"unknown quantization {quant!r}; using q4 (0.5 B/param). "
            f"Known: {', '.join(sorted(BYTES_PER_PARAM))}"
        )
        quant = "q4"
    if context_length <= 0:
        context_length = 8192
        warnings.append("context_length must be > 0; used 8192")
    parallel_requests = max(int(parallel_requests), 1)

    weights = model_size_b * 1e9 * BYTES_PER_PARAM[quant]
    # runtime overhead: context, activations, sampler, driver/allocator slack
    overhead = max(0.5 * GIB, 0.05 * weights)
    assumptions.append("runtime overhead = max(0.5 GiB, 5% of weights)")

    kv_pt, defaulted = kv_bytes_per_token(architecture)
    if defaulted:
        assumptions.append(
            "KV cache assumed 32 layers / 8 KV heads / head_dim 128 / fp16 "
            "(architecture unknown — pass 'architecture' for a better estimate)"
        )
    else:
        assumptions.append(f"KV cache from architecture '{architecture}'")
    kv_per_request = kv_pt * context_length * parallel_requests

    total = weights + overhead + kv_per_request
    total_gib = total / GIB

    vram_total_gib = vram_total_bytes / GIB if vram_total_bytes else None
    vram_free_gib = vram_free_bytes / GIB if vram_free_bytes else None

    fits: bool | None = None
    headroom: float | None = None
    budget = vram_free_gib if vram_free_gib is not None else vram_total_gib
    if budget is not None:
        fits = total_gib <= budget
        headroom = budget - total_gib
        if fits and headroom < 1.0:
            warnings.append("fits, but headroom < 1 GiB — long generations may OOM")
    else:
        warnings.append("no VRAM information available; cannot decide fit")

    return FitEstimate(
        available=True,
        weights_gib=round(weights / GIB, 2),
        runtime_overhead_gib=round(overhead / GIB, 2),
        kv_cache_gib_per_request=round(kv_per_request / GIB, 2),
        total_gib=round(total_gib, 2),
        vram_total_gib=round(vram_total_gib, 2) if vram_total_gib else None,
        vram_free_gib=round(vram_free_gib, 2) if vram_free_gib is not None else None,
        fits=fits,
        headroom_gib=round(headroom, 2) if headroom is not None else None,
        assumptions=assumptions,
        warnings=warnings,
    )
