"""Discovery merging, error isolation and device lifecycle."""

from __future__ import annotations

from gpu_manager_mcp.discovery import Discovery
from tests.conftest import FakeBackend


def test_multiple_devices_discovered():
    d = Discovery(backends=[FakeBackend(gpu_ids=["a-0", "a-1"]), FakeBackend(gpu_ids=["b-0"])])
    devices = d.refresh()
    assert {x.gpu_id for x in devices} == {"a-0", "a-1", "b-0"}


def test_broken_backend_does_not_hide_others():
    d = Discovery(backends=[FakeBackend(fail_discover=True), FakeBackend(gpu_ids=["ok-0"])])
    devices = d.refresh()
    assert [x.gpu_id for x in devices] == ["ok-0"]
    assert "fake" in d.backend_errors
    assert "simulated backend crash" in d.backend_errors["fake"]


def test_duplicate_ids_are_disambiguated():
    d = Discovery(backends=[FakeBackend(gpu_ids=["x"]), FakeBackend(gpu_ids=["x"])])
    ids = {x.gpu_id for x in d.refresh()}
    assert ids == {"x", "x#2"}


def test_device_disappears_on_refresh():
    backend = FakeBackend(gpu_ids=["live", "ghost"])
    d = Discovery(backends=[backend])
    assert d.device("ghost") is not None
    backend.gpu_ids = ["live"]  # device vanishes
    d.refresh()
    assert d.device("ghost") is None
    assert d.device("live") is not None


def test_unknown_id_returns_none():
    d = Discovery(backends=[FakeBackend()])
    assert d.device("nope") is None
