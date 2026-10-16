"""Command line entry point.

gpu-manager-mcp            # stdio MCP server (default)
gpu-manager-mcp summary    # print a one-shot system summary
gpu-manager-mcp http       # streamable-http on loopback
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import Config
from .server import GpuManager, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gpu-manager-mcp", description="Read-only GPU monitoring MCP server"
    )
    parser.add_argument("mode", nargs="?", default="stdio", choices=["stdio", "http", "summary"])
    parser.add_argument("--version", action="version", version=f"gpu-manager-mcp {__version__}")
    args = parser.parse_args(argv)

    if args.mode == "summary":
        mgr = GpuManager(Config())
        lines = []
        for info in mgr.discovery.devices():
            try:
                snap = mgr.collector.snapshot(info.gpu_id)
                lines.append(
                    {
                        "gpu_id": info.gpu_id,
                        "name": info.name,
                        "vram_total_bytes": info.vram_total_bytes,
                        "utilization": snap.utilization_percent.model_dump(),
                        "temperature_celsius": snap.temperature_celsius.model_dump(),
                    }
                )
            except Exception as exc:
                lines.append({"gpu_id": info.gpu_id, "error": str(exc)})
        json.dump(
            {"gpus": lines, "backend_errors": mgr.discovery.backend_errors}, sys.stdout, indent=2
        )
        sys.stdout.write("\n")
        return 0

    config = Config()
    if args.mode == "http":
        config.http_enabled = True
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
