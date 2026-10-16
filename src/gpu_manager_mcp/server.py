"""GPU Manager MCP server.

Default transport: stdio. Optional Streamable HTTP bound to loopback only.
The server is strictly read-only: no shell tool, no clock/fan/voltage
changes, no process control.
"""

from __future__ import annotations

import json
import threading
import time

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .config import Config
from .discovery import Discovery
from .inference import estimate_fit, find_candidates, rank_gpus
from .models import Capability
from .monitoring import Collector, HistoryStore, bottleneck_report, health_report


class GpuManager:
    """Wires discovery + collector + history together; shared by server & CLI."""

    def __init__(self, config: Config | None = None, backends: list | None = None):
        self.config = config or Config()
        problems = self.config.validate()
        if problems:
            raise ValueError("invalid configuration: " + "; ".join(problems))
        self.discovery = Discovery(self.config, backends=backends)
        self.collector = Collector(self.discovery, self.config.cache_ttl_seconds)
        self.history = HistoryStore(
            self.config.history_db_path,
            max_rows=self.config.history_max_rows,
            retention_hours=self.config.history_retention_hours,
        )
        self._sampler_stop = threading.Event()
        self._sampler: threading.Thread | None = None

    # -- background sampling ------------------------------------------------
    def start_sampler(self) -> None:
        if self._sampler is not None:
            return

        def loop() -> None:
            while not self._sampler_stop.wait(self.config.sample_interval_seconds):
                try:
                    for snap in self.collector.all_snapshots().values():
                        self.history.record(snap)
                except Exception:  # noqa: S112  sampling must never crash the server
                    continue

        self._sampler = threading.Thread(target=loop, name="gpu-sampler", daemon=True)
        self._sampler.start()

    def stop_sampler(self) -> None:
        self._sampler_stop.set()

    # -- helpers -------------------------------------------------------------
    def _require_device(self, gpu_id: str):
        info = self.discovery.device(gpu_id)
        if info is None:
            valid = [d.gpu_id for d in self.discovery.devices()]
            raise ToolError(f"unknown gpu_id {gpu_id!r}. Valid ids: {json.dumps(valid)}")
        return info

    def _snapshot(self, gpu_id: str):
        self._require_device(gpu_id)
        return self.collector.snapshot(gpu_id)


def build_server(config: Config | None = None) -> tuple[MCPServer, GpuManager]:
    config = config or Config()
    mgr = GpuManager(config)
    mcp = MCPServer(
        "gpu-manager-mcp",
        instructions=(
            "Read-only GPU monitoring for NVIDIA, AMD and Intel devices. "
            "Call list_gpus() first to obtain gpu_id values; every other tool "
            "takes one. Unsupported metrics are reported as unavailable with "
            "a reason — never as zero."
        ),
    )

    # -- discovery -----------------------------------------------------------
    @mcp.tool()
    def list_gpus() -> dict:
        """Discover all GPUs (NVIDIA/AMD/Intel). Returns stable gpu_ids, names,
        VRAM totals, PCI bus ids, driver versions and supported telemetry
        capabilities per device."""
        devices = mgr.discovery.refresh()
        return {
            "count": len(devices),
            "gpus": [d.model_dump(mode="json") for d in devices],
            "backend_errors": mgr.discovery.backend_errors,
        }

    @mcp.tool()
    def get_gpu_info(gpu_id: str) -> dict:
        """Hardware and driver information for one GPU."""
        return mgr._require_device(gpu_id).model_dump(mode="json")

    @mcp.tool()
    def get_system_gpu_summary() -> dict:
        """System-wide overview: every GPU with one-line key metrics."""
        lines = []
        for info in mgr.discovery.devices():
            try:
                snap = mgr.collector.snapshot(info.gpu_id)
            except Exception as exc:
                lines.append({"gpu_id": info.gpu_id, "name": info.name, "error": str(exc)})
                continue

            def fmt(mv):
                return f"{mv.value:g} {mv.unit}" if mv.available else f"n/a ({mv.reason})"

            lines.append(
                {
                    "gpu_id": info.gpu_id,
                    "name": info.name,
                    "manufacturer": info.manufacturer.value,
                    "utilization": fmt(snap.utilization_percent),
                    "vram_used": fmt(snap.memory.used_bytes),
                    "vram_total": fmt(snap.memory.total_bytes),
                    "temperature": fmt(snap.temperature_celsius),
                    "power": fmt(snap.power_watts),
                }
            )
        return {"gpus": lines, "backend_errors": mgr.discovery.backend_errors}

    # -- monitoring ----------------------------------------------------------
    def _metric(gpu_id: str, attr: str) -> dict:
        snap = mgr._snapshot(gpu_id)
        return getattr(snap, attr).model_dump(mode="json")

    @mcp.tool()
    def get_gpu_utilization(gpu_id: str) -> dict:
        """GPU core utilization percentage for one GPU."""
        return _metric(gpu_id, "utilization_percent")

    @mcp.tool()
    def get_gpu_memory(gpu_id: str) -> dict:
        """VRAM total/used/free in bytes for one GPU."""
        mgr._require_device(gpu_id)
        return mgr._snapshot(gpu_id).memory.model_dump(mode="json")

    @mcp.tool()
    def get_gpu_temperature(gpu_id: str) -> dict:
        """GPU temperature in Celsius (edge sensor where available)."""
        return _metric(gpu_id, "temperature_celsius")

    @mcp.tool()
    def get_gpu_power(gpu_id: str) -> dict:
        """GPU package power draw in watts."""
        return _metric(gpu_id, "power_watts")

    @mcp.tool()
    def get_gpu_clocks(gpu_id: str) -> dict:
        """Current graphics clock in MHz."""
        return _metric(gpu_id, "clock_mhz")

    @mcp.tool()
    def get_gpu_processes(gpu_id: str) -> dict:
        """Processes using the GPU with per-process VRAM, when the driver
        exposes it (NVML or rocm-smi). Returns supported=false otherwise."""
        info = mgr._require_device(gpu_id)
        backend = mgr.discovery.backend_for(gpu_id)
        if backend is None or Capability.PROCESSES not in info.capabilities:
            return {
                "gpu_id": gpu_id,
                "supported": False,
                "reason": "this backend/driver combination does not expose per-process GPU usage",
                "processes": [],
            }
        return {
            "gpu_id": gpu_id,
            "supported": True,
            "processes": [p.model_dump(mode="json") for p in backend.processes(gpu_id)],
        }

    @mcp.tool()
    def get_all_gpu_metrics() -> dict:
        """Full telemetry snapshot for every GPU at once."""
        snaps = mgr.collector.all_snapshots()
        return {
            "timestamp": time.time(),
            "snapshots": [s.model_dump(mode="json") for s in snaps.values()],
        }

    # -- history -------------------------------------------------------------
    @mcp.tool()
    def get_gpu_history(gpu_id: str, metric: str, duration_seconds: float = 300) -> dict:
        """Historical samples for one metric over a time window.
        Metrics: utilization_percent, memory_used_bytes, memory_total_bytes,
        memory_free_bytes, temperature_celsius, power_watts, clock_mhz."""
        mgr._require_device(gpu_id)
        duration_seconds = max(1.0, min(float(duration_seconds), 86_400.0))
        rows = mgr.history.query(gpu_id, metric, duration_seconds)
        return {
            "gpu_id": gpu_id,
            "metric": metric,
            "duration_seconds": duration_seconds,
            "samples": rows,
            "note": "history fills as the server samples; "
            "the sampler runs automatically inside the server process",
        }

    @mcp.tool()
    def get_gpu_peak_usage(
        gpu_id: str, metric: str = "utilization_percent", duration_seconds: float = 3600
    ) -> dict:
        """Max/min/avg of a metric over a window."""
        mgr._require_device(gpu_id)
        duration_seconds = max(1.0, min(float(duration_seconds), 86_400.0))
        peak = mgr.history.peak(gpu_id, metric, duration_seconds)
        if peak is None:
            return {
                "gpu_id": gpu_id,
                "metric": metric,
                "available": False,
                "reason": "no samples stored for this metric/window",
            }
        return {"gpu_id": gpu_id, "available": True, **peak}

    @mcp.tool()
    def get_gpu_health(gpu_id: str) -> dict:
        """Health check: thermal/memory/power issues from live + historical data."""
        info = mgr._require_device(gpu_id)
        snap = mgr._snapshot(gpu_id)
        return health_report(info, snap, mgr.history)

    # -- AI-specific ---------------------------------------------------------
    @mcp.tool()
    def estimate_model_fit(
        gpu_id: str,
        model_size_b: float,
        quantization: str = "q4",
        context_length: int = 8192,
        parallel_requests: int = 1,
        architecture: str | None = None,
    ) -> dict:
        """Estimate whether an LLM fits in one GPU's VRAM. Rough estimate:
        weights + runtime overhead + KV cache. Pass 'architecture' (e.g.
        llama-3-8b, mistral-7b) for an architecture-aware KV cache; without
        it, default GQA assumptions are used and flagged."""
        info = mgr._require_device(gpu_id)
        snap = mgr._snapshot(gpu_id)
        free = snap.memory.free_bytes.value if snap.memory.free_bytes.available else None
        total = info.vram_total_bytes or (
            snap.memory.total_bytes.value if snap.memory.total_bytes.available else None
        )
        est = estimate_fit(
            model_size_b,
            quantization,
            context_length,
            parallel_requests,
            architecture,
            total,
            free,
        )
        return {"gpu_id": gpu_id, **est.model_dump()}

    @mcp.tool()
    def recommend_gpu_for_model(
        model_size_b: float,
        quantization: str = "q4",
        context_length: int = 8192,
        parallel_requests: int = 1,
        architecture: str | None = None,
    ) -> dict:
        """Rank all discovered GPUs by estimated fit for the given model."""
        snaps = mgr.collector.all_snapshots()
        return {
            "ranking": rank_gpus(
                mgr.discovery.devices(),
                snaps,
                model_size_b,
                quantization,
                context_length,
                parallel_requests,
                architecture,
            )
        }

    @mcp.tool()
    def find_available_gpu(required_vram_gb: float) -> dict:
        """GPUs with at least required_vram_gb (GiB) free right now."""
        if required_vram_gb <= 0:
            raise ToolError("required_vram_gb must be > 0")
        return {
            "required_gb": required_vram_gb,
            "candidates": find_candidates(
                mgr.discovery.devices(), mgr.collector.all_snapshots(), required_vram_gb
            ),
        }

    @mcp.tool()
    def compare_gpus(gpu_ids: list[str]) -> dict:
        """Side-by-side comparison of specs and live telemetry for several GPUs."""
        out = []
        for gid in gpu_ids[:16]:
            info = mgr._require_device(gid)
            snap = mgr.collector.snapshot(gid)
            out.append(
                {
                    "info": info.model_dump(mode="json"),
                    "snapshot": snap.model_dump(mode="json"),
                }
            )
        return {"gpus": out}

    @mcp.tool()
    def detect_gpu_bottleneck(gpu_id: str, window_seconds: float = 300) -> dict:
        """Heuristic bottleneck analysis (compute/thermal/power/memory) from
        sampled history. Reports when evidence is insufficient."""
        info = mgr._require_device(gpu_id)
        snap = mgr._snapshot(gpu_id)
        window_seconds = max(10.0, min(float(window_seconds), 86_400.0))
        return bottleneck_report(info, snap, mgr.history, window_seconds)

    return mcp, mgr


def run(config: Config | None = None) -> None:
    config = config or Config()
    mcp, mgr = build_server(config)
    mgr.start_sampler()
    if config.http_enabled:
        mcp.run(transport="streamable-http", host=config.http_host, port=config.http_port)
    else:
        mcp.run(transport="stdio")
