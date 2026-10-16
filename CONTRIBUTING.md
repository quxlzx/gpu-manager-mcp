# Contributing

Thanks for your interest! GPU Manager MCP is read-only by design — keep it
that way.

## Ground rules

- **No hardware mutation.** Clocks, voltages, fan curves and process control
  are explicitly out of scope.
- **No arbitrary command execution.** External helpers (`nvidia-smi`,
  `rocm-smi`, `intel_gpu_top`) run only from the fixed allowlist in
  `config.py`, as argument arrays with timeouts.
- **Honest telemetry.** A sensor you cannot read is reported as
  `available: false` with a reason — never zero, never guessed.
- New vendors implement `GPUBackend` (`backends/base.py`) and register in
  `discovery.py`. The MCP layer should not need changes.

## Workflow

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
pytest -q                 # all tests must pass (mocked hardware; no GPU needed)
ruff check src tests && ruff format --check src tests
mypy src
```

Every new tool needs at least: a happy-path test, an unavailable-sensor
test, and a bad-input test.

## Releases

Semantic versioning. Tag releases as `vMAJOR.MINOR.PATCH`; release
candidates use `v0.1.0rc1` style. Update `__init__.py`, `pyproject.toml`
and the changelog in the release PR.
