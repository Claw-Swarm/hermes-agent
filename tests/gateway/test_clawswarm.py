"""Tests for the ClawSwarm gateway adapter."""

import os
import pytest
from unittest.mock import patch


def test_platform_enum_exists():
    from gateway.config import Platform
    assert Platform.CLAWSWARM.value == "clawswarm"


def test_config_loading_from_env():
    from gateway.config import Platform, GatewayConfig, _apply_env_overrides
    env = {
        "CLAWSWARM_TOKEN": "ocs_testtoken",
        "CLAWSWARM_SERVER_URL": "ws://localhost:3000",
        "CLAWSWARM_AGENT_NAME": "test-hermes",
    }
    with patch.dict(os.environ, env):
        config = GatewayConfig()
        _apply_env_overrides(config)
    assert Platform.CLAWSWARM in config.platforms
    pconfig = config.platforms[Platform.CLAWSWARM]
    assert pconfig.token == "ocs_testtoken"
    assert pconfig.extra["server_url"] == "ws://localhost:3000"
    assert pconfig.extra["agent_name"] == "test-hermes"


def test_check_requirements_missing_url(monkeypatch):
    from gateway.platforms.clawswarm import check_clawswarm_requirements
    monkeypatch.setenv("CLAWSWARM_TOKEN", "ocs_x")
    monkeypatch.setenv("CLAWSWARM_AGENT_NAME", "h")
    monkeypatch.delenv("CLAWSWARM_SERVER_URL", raising=False)
    assert check_clawswarm_requirements() is False


def test_check_requirements_missing_token(monkeypatch):
    from gateway.platforms.clawswarm import check_clawswarm_requirements
    monkeypatch.setenv("CLAWSWARM_SERVER_URL", "ws://localhost:3000")
    monkeypatch.setenv("CLAWSWARM_AGENT_NAME", "h")
    monkeypatch.delenv("CLAWSWARM_TOKEN", raising=False)
    assert check_clawswarm_requirements() is False


def test_check_requirements_ok(monkeypatch):
    from gateway.platforms.clawswarm import check_clawswarm_requirements
    monkeypatch.setenv("CLAWSWARM_SERVER_URL", "ws://localhost:3000")
    monkeypatch.setenv("CLAWSWARM_TOKEN", "ocs_x")
    monkeypatch.setenv("CLAWSWARM_AGENT_NAME", "hermes")
    assert check_clawswarm_requirements() is True


def test_authorization_maps():
    """ClawSwarm must appear in both auth maps in gateway/run.py."""
    import importlib
    import inspect
    run = importlib.import_module("gateway.run")
    src = inspect.getsource(run)
    assert "CLAWSWARM_ALLOWED_USERS" in src
    assert "CLAWSWARM_ALLOW_ALL_USERS" in src


def test_adapter_init(monkeypatch):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.clawswarm import ClawSwarmAdapter
    monkeypatch.setenv("CLAWSWARM_SERVER_URL", "ws://localhost:3000")
    monkeypatch.setenv("CLAWSWARM_TOKEN", "ocs_abc")
    monkeypatch.setenv("CLAWSWARM_AGENT_NAME", "myagent")
    cfg = PlatformConfig()
    adapter = ClawSwarmAdapter(cfg)
    assert adapter._server_url == "ws://localhost:3000"
    assert adapter._token == "ocs_abc"
    assert adapter._agent_name == "myagent"


def test_send_message_tool_has_clawswarm():
    import inspect
    from tools import send_message_tool as m
    src = inspect.getsource(m)
    assert "clawswarm" in src.lower()


def test_toolset_registered():
    from toolsets import TOOLSETS
    assert "hermes-clawswarm" in TOOLSETS
