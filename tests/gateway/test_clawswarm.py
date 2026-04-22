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


@pytest.mark.asyncio
async def test_handle_raw_no_context(monkeypatch):
    """Messages without _context (e.g. sent from admin panel) must still be dispatched."""
    import asyncio
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.clawswarm import ClawSwarmAdapter

    monkeypatch.setenv("CLAWSWARM_SERVER_URL", "ws://localhost:3000")
    monkeypatch.setenv("CLAWSWARM_TOKEN", "ocs_abc")
    monkeypatch.setenv("CLAWSWARM_AGENT_NAME", "hermes")

    cfg = PlatformConfig()
    adapter = ClawSwarmAdapter(cfg)

    received = []

    async def fake_handle(event):
        received.append(event)

    adapter.handle_message = fake_handle

    msg = {
        "type": "message",
        "id": "msg1",
        "roomId": "room1",
        "from": "admin",
        "content": "hello hermes",
        # no _context field
    }
    await adapter._handle_raw(__import__("json").dumps(msg))

    assert len(received) == 1
    assert received[0].text == "hello hermes"
    assert received[0].extra["room_id"] == "room1"


@pytest.mark.asyncio
async def test_handle_raw_with_skills(monkeypatch):
    """_context with agentSkills must appear in extra_system."""
    from gateway.config import PlatformConfig
    from gateway.platforms.clawswarm import ClawSwarmAdapter

    monkeypatch.setenv("CLAWSWARM_SERVER_URL", "ws://localhost:3000")
    monkeypatch.setenv("CLAWSWARM_TOKEN", "ocs_abc")
    monkeypatch.setenv("CLAWSWARM_AGENT_NAME", "hermes")

    adapter = ClawSwarmAdapter(PlatformConfig())
    received = []

    async def fake_handle(event):
        received.append(event)

    adapter.handle_message = fake_handle

    import json as _json
    msg = {
        "type": "message",
        "id": "m1",
        "roomId": "r1",
        "from": "user1",
        "content": "hello",
        "_context": {
            "agentName": "hermes",
            "agentNickname": "",
            "agentRole": "assistant",
            "agentSkills": ["coding", "research"],
            "taskLabel": "test",
            "taskHistory": [],
            "roomAgents": [],
        },
    }
    await adapter._handle_raw(_json.dumps(msg))
    assert len(received) == 1
    assert "coding, research" in received[0].extra["extra_system"]


@pytest.mark.asyncio
async def test_handle_raw_empty_content_dropped(monkeypatch):
    """Messages with empty/whitespace content must be silently dropped."""
    from gateway.config import PlatformConfig
    from gateway.platforms.clawswarm import ClawSwarmAdapter

    monkeypatch.setenv("CLAWSWARM_SERVER_URL", "ws://localhost:3000")
    monkeypatch.setenv("CLAWSWARM_TOKEN", "ocs_abc")
    monkeypatch.setenv("CLAWSWARM_AGENT_NAME", "hermes")

    adapter = ClawSwarmAdapter(PlatformConfig())
    received = []

    async def fake_handle(event):
        received.append(event)

    adapter.handle_message = fake_handle

    import json as _json
    msg = {"type": "message", "id": "m2", "roomId": "r1", "from": "user1", "content": "   "}
    await adapter._handle_raw(_json.dumps(msg))
    assert len(received) == 0


@pytest.mark.asyncio
async def test_handle_raw_missing_room_id_dropped(monkeypatch):
    """Messages without roomId must be silently dropped."""
    from gateway.config import PlatformConfig
    from gateway.platforms.clawswarm import ClawSwarmAdapter

    monkeypatch.setenv("CLAWSWARM_SERVER_URL", "ws://localhost:3000")
    monkeypatch.setenv("CLAWSWARM_TOKEN", "ocs_abc")
    monkeypatch.setenv("CLAWSWARM_AGENT_NAME", "hermes")

    adapter = ClawSwarmAdapter(PlatformConfig())
    received = []

    async def fake_handle(event):
        received.append(event)

    adapter.handle_message = fake_handle

    import json as _json
    msg = {"type": "message", "id": "m3", "from": "user1", "content": "hi"}
    await adapter._handle_raw(_json.dumps(msg))
    assert len(received) == 0


@pytest.mark.asyncio
async def test_auth_error_stops_reconnect(monkeypatch):
    """auth_error frame must set _closing=True to prevent reconnect spam."""
    from gateway.config import PlatformConfig
    from gateway.platforms.clawswarm import ClawSwarmAdapter

    monkeypatch.setenv("CLAWSWARM_SERVER_URL", "ws://localhost:3000")
    monkeypatch.setenv("CLAWSWARM_TOKEN", "ocs_bad")
    monkeypatch.setenv("CLAWSWARM_AGENT_NAME", "hermes")

    adapter = ClawSwarmAdapter(PlatformConfig())
    import json as _json
    await adapter._handle_raw(_json.dumps({"type": "auth_error", "message": "invalid token"}))
    assert adapter._closing is True
