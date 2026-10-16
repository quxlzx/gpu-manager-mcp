"""End-to-end: a real MCP client over stdio against the real server."""

from __future__ import annotations

import contextlib
import json
import os
import sys

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

TOOL_NAMES = {
    "list_gpus",
    "get_gpu_info",
    "get_system_gpu_summary",
    "get_gpu_utilization",
    "get_gpu_memory",
    "get_gpu_temperature",
    "get_gpu_power",
    "get_gpu_clocks",
    "get_gpu_processes",
    "get_all_gpu_metrics",
    "get_gpu_history",
    "get_gpu_peak_usage",
    "get_gpu_health",
    "estimate_model_fit",
    "recommend_gpu_for_model",
    "find_available_gpu",
    "compare_gpus",
    "detect_gpu_bottleneck",
}


@contextlib.asynccontextmanager
async def open_session(tmp_path):
    """Full client lifecycle inside one task (anyio cancel-scope requirement)."""
    env = dict(os.environ, GPU_MANAGER_DB_PATH=str(tmp_path / "h.sqlite3"))
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "gpu_manager_mcp.cli", "stdio"],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            yield s


async def test_initialize_and_tool_discovery(tmp_path):
    async with open_session(tmp_path) as s:
        tools = await s.list_tools()
        names = {t.name for t in tools.tools}
        assert TOOL_NAMES <= names, TOOL_NAMES - names
        assert all(t.description for t in tools.tools)


async def test_list_gpus_call(tmp_path):
    async with open_session(tmp_path) as s:
        result = await s.call_tool("list_gpus", {})
        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert "count" in payload and "gpus" in payload


async def test_unknown_gpu_id_returns_error_result(tmp_path):
    async with open_session(tmp_path) as s:
        result = await s.call_tool("get_gpu_temperature", {"gpu_id": "does-not-exist; rm -rf /"})
        assert result.is_error
        assert "unknown gpu_id" in result.content[0].text


async def test_find_available_gpu_validation(tmp_path):
    async with open_session(tmp_path) as s:
        result = await s.call_tool("find_available_gpu", {"required_vram_gb": -5})
        assert result.is_error


async def test_estimate_model_fit_shape(tmp_path):
    async with open_session(tmp_path) as s:
        listing = await s.call_tool("list_gpus", {})
        payload = json.loads(listing.content[0].text)
        if not payload["gpus"]:
            pytest.skip("no GPUs on this host")
        gid = payload["gpus"][0]["gpu_id"]
        result = await s.call_tool(
            "estimate_model_fit",
            {
                "gpu_id": gid,
                "model_size_b": 8,
                "quantization": "q4",
                "context_length": 8192,
                "parallel_requests": 1,
            },
        )
        assert not result.is_error
        est = json.loads(result.content[0].text)
        assert {"weights_gib", "total_gib", "assumptions"} <= set(est)
