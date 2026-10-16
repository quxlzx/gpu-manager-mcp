# GPU Manager MCP

Read-only GPU monitoring and AI-workload planning over the
[Model Context Protocol](https://modelcontextprotocol.io). One stdio server,
any MCP client, no LLM required.

Discover NVIDIA, AMD and Intel GPUs, watch utilization / VRAM / temperature /
power / clocks, keep bounded local history in SQLite, and estimate whether
your LLM actually fits before you load it.

- **Cross-platform backends** — NVML (via `pynvml`) and `nvidia-smi` for
  NVIDIA; amdgpu sysfs (no ROCm needed) and `rocm-smi` fallback for AMD;
  i915/xe sysfs for Intel.
- **Honest telemetry** — every reading carries a timestamp, unit and source.
  Unsupported sensors are reported as `available: false` **with a reason**,
  never silently zeroed.
- **Read-only by design** — no shell tool, no overclocking, no voltage, no
  fan control, no process termination. External helpers run from a fixed
  allowlist with timeouts and no shell interpolation.
- **AI-aware** — `estimate_model_fit()` computes weights + runtime overhead +
  KV-cache needs for a given quantization, context length and parallel
  request count, then checks it against live free VRAM.

## Installation

```bash
pip install gpu-manager-mcp            # (once published; see below)
# or from source
git clone https://github.com/quxlzx/gpu-manager-mcp
cd gpu-manager-mcp
python -m venv .venv && .venv/bin/pip install -e .
```

Optional NVIDIA NVML bindings (recommended on NVIDIA hosts):

```bash
pip install "gpu-manager-mcp[nvidia]"   # or: pip install nvidia-ml-py
```

Requirements: Python 3.11+. Linux is fully supported; on Windows the AMD
backend is not yet implemented (see Known limitations).

## Supported GPUs

| Vendor | Path | Util | VRAM | Temp | Power | Clocks | Processes |
|---|---|---|---|---|---|---|---|
| NVIDIA (GeForce/Quadro/RTX/Tesla) | NVML (`pynvml`) | ✅ | ✅ | ✅ | ✅* | ✅* | ✅* |
| NVIDIA (fallback) | `nvidia-smi` CSV | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| AMD (Linux, amdgpu driver) | sysfs + hwmon | ✅* | ✅ | ✅* | ✅* | ✅* | via `rocm-smi` |
| Intel (Linux, i915/xe) | sysfs + hwmon | ✅* | ❌ | ✅* | ✅* | ✅* | ❌ |

`✅*` = depends on card/driver; unavailable sensors are reported with the
driver's own error, not guessed.

## Connecting an MCP client

Generic stdio configuration (any MCP client):

```json
{
  "mcpServers": {
    "gpu-manager": {
      "command": "/path/to/.venv/bin/gpu-manager-mcp",
      "args": ["stdio"]
    }
  }
}
```

Hermes Agent (`~/.hermes/config.yaml`):

```yaml
mcp_servers:
  gpu-manager:
    command: /path/to/.venv/bin/gpu-manager-mcp
    args: ["stdio"]
    timeout: 60
```

Claude Desktop / Cursor / Cline: same `command` + `args` shape as above.

## Tools

Discovery: `list_gpus`, `get_gpu_info`, `get_system_gpu_summary`
Monitoring: `get_gpu_utilization`, `get_gpu_memory`, `get_gpu_temperature`,
`get_gpu_power`, `get_gpu_clocks`, `get_gpu_processes`,
`get_all_gpu_metrics`
History: `get_gpu_history`, `get_gpu_peak_usage`, `get_gpu_health`
AI planning: `estimate_model_fit`, `recommend_gpu_for_model`,
`find_available_gpu`, `compare_gpus`, `detect_gpu_bottleneck`

Always call `list_gpus()` first — every other tool takes a `gpu_id` from it.

### Example

`estimate_model_fit(gpu_id="amd-0000:03:00.0", model_size_b=7,
quantization="q4", context_length=4096, parallel_requests=1,
architecture="qwen2-7b")` — real output from an 8 GiB RX 6600 XT:

```json
{
  "gpu_id": "amd-0000:03:00.0",
  "available": true,
  "weights_gib": 3.26,
  "runtime_overhead_gib": 0.5,
  "kv_cache_gib_per_request": 0.22,
  "total_gib": 3.98,
  "vram_total_gib": 7.98,
  "vram_free_gib": 7.97,
  "fits": true,
  "headroom_gib": 3.99,
  "assumptions": ["runtime overhead = max(0.5 GiB, 5% of weights)",
                   "KV cache from architecture 'qwen2-7b'"],
  "warnings": []
}
```

A metric that the hardware cannot provide looks like:

```json
{
  "available": false,
  "reason": "sensor busy — the GPU power management interface refused the read (common when the card is in a low-power state)",
  "source": "amd-sysfs"
}
```

## Configuration

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `GPU_MANAGER_SAMPLE_INTERVAL` | `2.0` | background sampling period (s) |
| `GPU_MANAGER_HISTORY_RETENTION_HOURS` | `24` | history window kept |
| `GPU_MANAGER_HISTORY_MAX_ROWS` | `200000` | hard row cap (bounded storage) |
| `GPU_MANAGER_DB_PATH` | `~/.local/share/gpu-manager-mcp/history.sqlite3` | SQLite location |
| `GPU_MANAGER_CACHE_TTL` | `1.0` | telemetry cache TTL (s) for <250 ms repeat queries |
| `GPU_MANAGER_COMMAND_TIMEOUT` | `10` | timeout for allowlisted helper CLIs |
| `GPU_MANAGER_HTTP_ENABLED` | `false` | enable Streamable HTTP transport |
| `GPU_MANAGER_HTTP_HOST` / `_PORT` | `127.0.0.1` / `8940` | loopback-only HTTP bind |

## Security guidance

- The server exposes **monitoring only**. There is no tool that executes
  arbitrary commands, changes clocks/fans/voltages, or kills processes.
- Helper CLIs (`nvidia-smi`, `rocm-smi`, `intel_gpu_top`) are invoked from a
  fixed allowlist with argument arrays and timeouts — never via a shell and
  never with user-supplied strings.
- `gpu_id` values are validated against discovered devices; unknown or
  crafted ids are rejected with a clean error.
- The optional HTTP transport binds to loopback by default and refuses
  non-loopback hosts unless you explicitly change it — and you should not,
  without adding authentication in front of it. Do not expose this server to
  the public internet.
- History is local SQLite with a bounded row cap; nothing leaves the machine.

## Known limitations

- **Windows**: AMD telemetry via sysfs is Linux-only; a Windows backend is
  planned (AMD-ADL / PDH or `amdsmi` bindings). The server runs on Windows
  but will report no AMD devices.
- **AMD hwmon reads can return EBUSY** when the card is in a low-power
  state; the reading is reported as unavailable with that explanation.
- **Intel VRAM** is not exposed by i915/xe sysfs; memory metrics are
  reported unavailable rather than estimated.
- **KV-cache estimates** are architecture-dependent. Without an
  `architecture` hint the server uses documented default assumptions and
  flags them. Treat all fit numbers as engineering estimates.
- Per-process VRAM requires NVML or `rocm-smi`; plain sysfs cannot provide it.

## Troubleshooting

- **No GPUs found on Linux** — check `/sys/class/drm/card*/device/vendor`
  exists and the driver is `amdgpu`/`i915`/`xe`; NVIDIA needs the proprietary
  driver + `nvidia-ml-py` or `nvidia-smi` on PATH.
- **`sensor busy` on AMD** — normal at idle on some cards; retry under load
  or check `power_state` in sysfs.
- **Permission denied** — NVML on some setups needs root or
  `CAP_SYS_ADMIN`; sysfs sensors are world-readable on standard distros.
- **History empty** — the sampler runs inside the server process; give it a
  few sample intervals, or call `get_all_gpu_metrics()` to force samples.

## Development

```bash
pip install -e ".[dev]"
pytest                 # unit + MCP client tests (mocked hardware)
ruff check src tests
mypy src
python -m build        # package validation
```

## License

MIT — see [LICENSE](LICENSE).
