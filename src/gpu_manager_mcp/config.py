"""Configuration for the GPU Manager MCP server.

All values may be overridden with ``GPU_MANAGER_*`` environment variables so
MCP clients that cannot pass CLI flags (stdio launchers) can still configure
the server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_TRUE = {"1", "true", "yes", "on"}


def _env(name: str, default):
    raw = os.environ.get(f"GPU_MANAGER_{name}")
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in _TRUE
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Config:
    # History / sampling
    sample_interval_seconds: float = field(
        default_factory=lambda: float(_env("SAMPLE_INTERVAL", 2.0))
    )
    history_retention_hours: float = field(
        default_factory=lambda: float(_env("HISTORY_RETENTION_HOURS", 24.0))
    )
    history_max_rows: int = field(default_factory=lambda: int(_env("HISTORY_MAX_ROWS", 200_000)))
    history_db_path: str = field(
        default_factory=lambda: str(
            _env(
                "DB_PATH",
                str(Path.home() / ".local" / "share" / "gpu-manager-mcp" / "history.sqlite3"),
            )
        )
    )
    # Telemetry cache: keeps repeated tool calls fast (<250 ms target) without
    # hammering the driver.
    cache_ttl_seconds: float = field(default_factory=lambda: float(_env("CACHE_TTL", 1.0)))
    # External helper commands (nvidia-smi, rocm-smi, intel_gpu_top) are only
    # ever invoked from this fixed allowlist, never from user input.
    allowed_commands: dict[str, list[str]] = field(
        default_factory=lambda: {
            "nvidia-smi": ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            "rocm-smi": ["rocm-smi", "--showname"],
            "intel_gpu_top": ["intel_gpu_top", "-J", "-l", "100", "-o", "-"],
        }
    )
    command_timeout_seconds: float = field(
        default_factory=lambda: float(_env("COMMAND_TIMEOUT", 10.0))
    )
    # Optional HTTP transport (off by default; stdio is the default transport).
    http_enabled: bool = field(default_factory=lambda: bool(_env("HTTP_ENABLED", False)))
    http_host: str = field(default_factory=lambda: str(_env("HTTP_HOST", "127.0.0.1")))
    http_port: int = field(default_factory=lambda: int(_env("HTTP_PORT", 8940)))

    def validate(self) -> list[str]:
        """Return a list of configuration problems (empty = OK)."""
        problems: list[str] = []
        if self.sample_interval_seconds < 0.1:
            problems.append("sample_interval_seconds must be >= 0.1")
        if self.cache_ttl_seconds < 0:
            problems.append("cache_ttl_seconds must be >= 0")
        if self.history_max_rows < 100:
            problems.append("history_max_rows must be >= 100")
        if self.history_retention_hours <= 0:
            problems.append("history_retention_hours must be > 0")
        if self.command_timeout_seconds <= 0:
            problems.append("command_timeout_seconds must be > 0")
        if self.http_enabled and self.http_host not in ("127.0.0.1", "localhost", "::1"):
            problems.append("http_host must be a loopback address unless you add authentication")
        return problems
