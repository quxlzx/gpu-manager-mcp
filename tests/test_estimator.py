"""Memory-fit estimation math and honesty about assumptions."""

from __future__ import annotations

import pytest

from gpu_manager_mcp.inference.estimator import GIB, estimate_fit, kv_bytes_per_token


def test_q4_weights_8b():
    est = estimate_fit(8, "q4", 4096, 1, "llama-3-8b", 8 * GIB, 8 * GIB)
    assert est.weights_gib == pytest.approx(8 * 1e9 * 0.5 / GIB, abs=0.05)  # ~3.73
    assert est.fits is True


def test_quantization_scales_weights():
    base = estimate_fit(70, "q4", 8192, 1, "llama-3-70b")
    for q, factor in (("q8", 2.0), ("q5", 1.25), ("q6", 1.5), ("f16", 4.0)):
        other = estimate_fit(70, q, 8192, 1, "llama-3-70b")
        assert other.weights_gib == pytest.approx(base.weights_gib * factor, rel=0.02)


def test_context_length_scales_kv():
    a = estimate_fit(8, "q4", 4096, 1, "llama-3-8b")
    b = estimate_fit(8, "q4", 8192, 1, "llama-3-8b")
    assert b.kv_cache_gib_per_request == pytest.approx(2 * a.kv_cache_gib_per_request, rel=0.01)


def test_parallel_requests_scale_kv():
    a = estimate_fit(8, "q4", 8192, 1, "llama-3-8b")
    b = estimate_fit(8, "q4", 8192, 4, "llama-3-8b")
    assert b.kv_cache_gib_per_request == pytest.approx(4 * a.kv_cache_gib_per_request, rel=0.01)


def test_insufficient_vram_fits_false():
    est = estimate_fit(70, "q4", 8192, 1, "llama-3-70b", 8 * GIB, 8 * GIB)
    assert est.fits is False
    assert est.headroom_gib < 0


def test_unknown_architecture_is_flagged():
    est = estimate_fit(13, "q4", 8192, 1, None)
    assert any("architecture" in a for a in est.assumptions)
    est2 = estimate_fit(13, "q4", 8192, 1, "mistral-7b")
    assert not any("architecture unknown" in a for a in est2.assumptions)


def test_unknown_quantization_warns_and_defaults():
    est = estimate_fit(8, "banana", 8192, 1)
    assert any("unknown quantization" in w for w in est.warnings)
    assert est.weights_gib == pytest.approx(estimate_fit(8, "q4").weights_gib)


def test_no_vram_cannot_decide():
    est = estimate_fit(8, "q4", 8192, 1)
    assert est.fits is None
    assert any("no VRAM" in w for w in est.warnings)


def test_invalid_model_size():
    est = estimate_fit(0)
    assert est.available is False


def test_kv_arch_specific():
    pt_8b, _ = kv_bytes_per_token("llama-3-8b")
    pt_70b, _ = kv_bytes_per_token("llama-3-70b")
    assert pt_70b == pytest.approx(2.5 * pt_8b)  # 80 vs 32 layers, same kv heads


def test_tight_headroom_warns():
    est = estimate_fit(8, "q4", 8192, 1, "llama-3-8b", 6 * GIB, 6 * GIB)
    assert est.fits is True
    assert any("headroom" in w for w in est.warnings)
