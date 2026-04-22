# ClawSwarm Connector Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add ClawSwarm as a Hermes gateway platform — WebSocket client that connects to a ClawSwarm multi-agent group chat server and dispatches @-mentioned messages to the Hermes agent.

**Architecture:** Hermes acts as a WS client connecting to `{serverUrl}/ws/agent`. After authenticating, it listens for `message` events with `_context` (meaning Hermes was @-mentioned). It replies by sending `{type:"message", roomId, content}` back over the same WS connection. Reconnection uses exponential backoff identical to the Mattermost adapter pattern.

**Tech Stack:** Python asyncio + websockets library (already a dependency), following `gateway/platforms/mattermost.py` as the reference adapter.

**Reference files:**
- Protocol source: `~/.hermes/cache/documents/doc_a07366b4aee6_channel.js`
- Reference adapter: `gateway/platforms/mattermost.py`
- Integration guide: `gateway/platforms/ADDING_A_PLATFORM.md`

---

## Task 1: Core Adapter — `gateway/platforms/clawswarm.py`

**Objective:** Create the WS client adapter that handles connect/auth/receive/send/typing.

**File:** Create `gateway/platforms/clawswarm.py`

**Complete implementation:**

```python
"""ClawSwarm gateway adapter.

Connects Hermes to a ClawSwarm multi-agent group chat server via WebSocket.
Hermes acts as a WS client; it authenticates, then responds when @-mentioned.

Environment variables:
    CLAWSWARM_SERVER_URL       WebSocket base URL (e.g. ws://localhost:3000)
    CLAWSWARM_TOKEN            Auth token (starts with ocs_)
    CLAWSWARM_AGENT_NAME       Display name for this Hermes instance
    CLAWSWARM_ALLOWED_USERS    Comma-separated allowed user names (optional)
    CLAWSWARM_HOME_CHANNEL     Room ID for cron/notification delivery (optional)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from typing import Any, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

_RECONNECT_BASE_DELAY = 1.0
_RECONNECT_MAX_DELAY = 30.0
_RECONNECT_JITTER = 0.2


def check_clawswarm_requirements() -> bool:
    """Return True if ClawSwarm adapter dependencies are available."""
    if not os.getenv("CLAWSWARM_SERVER_URL"):
        logger.debug("ClawSwarm: CLAWSWARM_SERVER_URL not set")
        return False
    if not os.getenv("CLAWSWARM_TOKEN"):
        logger.debug("ClawSwarm: CLAWSWARM_TOKEN not set")
        return False
    if not os.getenv("CLAWSWARM_AGENT_NAME"):
        logger.debug("ClawSwarm: CLAWSWARM_AGENT_NAME not set")
        return False
    try:
        import websockets  # noqa: F401
        return True
    except ImportError:
        logger.warning("ClawSwarm: websockets not installed — run: pip install websockets")
        return False


class ClawSwarmAdapter(BasePlatformAdapter):
    """Gateway adapter for ClawSwarm multi-agent group chat."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.CLAWSWARM)

        self._server_url: str = (
            config.extra.get("server_url", "")
            or os.getenv("CLAWSWARM_SERVER_URL", "")
        ).rstrip("/")
        self._token: str = config.token or os.getenv("CLAWSWARM_TOKEN", "")
        self._agent_name: str = (
            config.extra.get("agent_name", "")
            or os.getenv("CLAWSWARM_AGENT_NAME", "hermes")
        )

        # Active WS connection handle
        self._ws: Any = None
        self._ws_task: Optional[asyncio.Task] = None
        self._closing = False

        # Current room ID (set on first inbound message)
        self._room_id: Optional[str] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Start WS connection loop in background task."""
        self._closing = False
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("ClawSwarm: connecting to %s as %s", self._server_url, self._agent_name)
        return True

    async def disconnect(self) -> None:
        """Gracefully stop the WS loop."""
        self._closing = True
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        logger.info("ClawSwarm: disconnected")

    # ── WebSocket loop ─────────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        """Connect, authenticate, receive messages; reconnect on failure."""
        import websockets

        delay = _RECONNECT_BASE_DELAY
        ws_url = f"{self._server_url}/ws/agent"

        while not self._closing:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws = ws
                    delay = _RECONNECT_BASE_DELAY  # reset on successful connect

                    # Authenticate
                    await ws.send(json.dumps({
                        "type": "auth",
                        "token": self._token,
                        "agentName": self._agent_name,
                    }))
                    logger.info("ClawSwarm: authenticated as %s", self._agent_name)

                    # Receive loop
                    async for raw in ws:
                        if self._closing:
                            break
                        await self._handle_raw(raw)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._closing:
                    break
                jitter = random.uniform(0, _RECONNECT_JITTER * delay)
                logger.warning(
                    "ClawSwarm: connection lost (%s) — reconnecting in %.1fs",
                    exc, delay + jitter,
                )
                await asyncio.sleep(delay + jitter)
                delay = min(delay * 2, _RECONNECT_MAX_DELAY)

        self._ws = None

    async def _handle_raw(self, raw: str | bytes) -> None:
        """Parse and dispatch an inbound WS message."""
        try:
            msg = json.loads(raw)
        except Exception:
            return

        if msg.get("type") != "message":
            return

        # Only process @-mention messages (have _context)
        if not msg.get("_context"):
            return

        self._room_id = msg.get("roomId")
        context = msg["_context"]

        # Build the text body to send to the agent
        content: str = msg.get("content", "")
        from_user: str = msg.get("from", "unknown")
        msg_id: str = msg.get("id", "")
        room_id: str = msg.get("roomId", "")

        # Strip /delegate @name or /discuss @name prefix, keep the actual task
        import re
        delegate_m = re.search(r'(?:^|\n)/delegate\s+@\S+\s+([\s\S]+)', content, re.IGNORECASE)
        discuss_m = re.search(r'(?:^|\n)/discuss\s+@\S+\s+([\s\S]+)', content, re.IGNORECASE)
        effective_body = (delegate_m or discuss_m)
        if effective_body:
            effective_body = effective_body.group(1).strip()
        else:
            effective_body = content

        # Build session key — one session per message (clean slate per mention)
        chat_id = f"{room_id}:{msg_id}"

        # Build system context lines from _context
        context_lines: list[str] = []

        agent_name = context.get("agentName") or self._agent_name
        agent_nickname = context.get("agentNickname", "")
        self_display = f"{agent_nickname} (@{agent_name})" if agent_nickname else f"@{agent_name}"
        context_lines.append(
            f"YOUR IDENTITY: You are {self_display}. This is your name in this group chat."
        )

        task_label = context.get("taskLabel", "")
        context_lines.append(
            f"Task scope: {f'\"{ task_label }\"' if task_label else '(no label)'}. "
            "Only use the conversation history provided as context."
        )

        if context.get("agentRole"):
            context_lines.append(f"Your role: {context['agentRole']}")

        other_agents = context.get("roomAgents", [])
        if other_agents:
            agent_list = []
            for a in other_agents:
                display = f"{a['nickname']} (@{a['name']})" if a.get("nickname") else f"@{a['name']}"
                agent_list.append(f"- {display}" + (f" — {a['role']}" if a.get("role") else ""))
            context_lines.append(
                "Other agents in this room:\n" + "\n".join(agent_list) + "\n\n"
                "HOW TO INTERACT WITH OTHER AGENTS:\n"
                "  /discuss @name <content> — ask a question or continue discussion\n"
                "  /delegate @name <task>   — formally hand off a task\n"
                "Commands must appear on their own line. Only ONE command per message."
            )

        task_history = context.get("taskHistory", [])
        if task_history:
            history_lines = ["## Conversation history (most recent last)"]
            for h in task_history:
                if h.get("id") == msg_id:
                    continue
                role = "Agent" if h.get("fromType") == "agent" else "Human"
                history_lines.append(f"[{role}] {h['from']}: {h['content']}")
            if len(history_lines) > 1:
                context_lines.append("\n".join(history_lines))

        extra_system = "\n\n".join(context_lines)

        source = self.build_source(
            chat_id=chat_id,
            chat_type="group",
            user_id=from_user,
            username=from_user,
            platform=Platform.CLAWSWARM,
        )

        event = MessageEvent(
            message_id=msg_id,
            chat_id=chat_id,
            text=effective_body,
            message_type=MessageType.TEXT,
            source=source,
            raw=msg,
            extra={"extra_system": extra_system, "room_id": room_id},
        )

        await self.handle_message(event)

    # ── Outbound ───────────────────────────────────────────────────────────

    async def send(self, chat_id: str, text: str, **kwargs) -> SendResult:
        """Send a reply to the current room."""
        room_id = kwargs.get("room_id") or self._room_id
        if not room_id:
            return SendResult(success=False, error="room_id unknown")
        if not self._ws:
            return SendResult(success=False, error="not connected")
        try:
            await self._ws.send(json.dumps({
                "type": "message",
                "roomId": room_id,
                "content": text,
            }))
            return SendResult(success=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, **kwargs) -> None:
        """Send typing indicator."""
        room_id = kwargs.get("room_id") or self._room_id
        if not room_id or not self._ws:
            return
        try:
            await self._ws.send(json.dumps({
                "type": "typing",
                "roomId": room_id,
                "status": True,
            }))
        except Exception:
            pass

    async def get_chat_info(self, chat_id: str) -> dict:
        room_id = chat_id.split(":")[0] if ":" in chat_id else chat_id
        return {"name": f"ClawSwarm room {room_id}", "type": "group", "chat_id": chat_id}
```

**Commit:**
```bash
cd ~/.hermes/hermes-agent
git checkout main && git pull
git checkout -b Hermes@clawswarm-connector
git add gateway/platforms/clawswarm.py
git commit -m "feat(gateway): add ClawSwarm WS client adapter"
```

---

## Task 2: Platform Enum — `gateway/config.py`

**Objective:** Register `CLAWSWARM` in the Platform enum and add env var loading.

**File:** Modify `gateway/config.py`

**Step 1:** Add to `Platform` enum (after `MATTERMOST`):
```python
CLAWSWARM = "clawswarm"
```

**Step 2:** Add to `get_connected_platforms()` token map (same section as MATTERMOST):
```python
Platform.CLAWSWARM: "CLAWSWARM_TOKEN",
```

**Step 3:** Add env var loader in `_apply_env_overrides()` (after mattermost block):
```python
# ClawSwarm
clawswarm_token = os.getenv("CLAWSWARM_TOKEN")
if clawswarm_token:
    clawswarm_url = os.getenv("CLAWSWARM_SERVER_URL", "")
    clawswarm_name = os.getenv("CLAWSWARM_AGENT_NAME", "hermes")
    if not clawswarm_url:
        logger.warning("CLAWSWARM_TOKEN set but CLAWSWARM_SERVER_URL is missing")
    if Platform.CLAWSWARM not in config.platforms:
        config.platforms[Platform.CLAWSWARM] = PlatformConfig()
    config.platforms[Platform.CLAWSWARM].enabled = True
    config.platforms[Platform.CLAWSWARM].token = clawswarm_token
    config.platforms[Platform.CLAWSWARM].extra["server_url"] = clawswarm_url
    config.platforms[Platform.CLAWSWARM].extra["agent_name"] = clawswarm_name
clawswarm_home = os.getenv("CLAWSWARM_HOME_CHANNEL")
if clawswarm_home and Platform.CLAWSWARM in config.platforms:
    config.platforms[Platform.CLAWSWARM].home_channel = HomeChannel(
        platform=Platform.CLAWSWARM,
        chat_id=clawswarm_home,
        name=os.getenv("CLAWSWARM_HOME_CHANNEL_NAME", "Home"),
    )
```

**Commit:**
```bash
git add gateway/config.py
git commit -m "feat(gateway): register ClawSwarm platform enum and env vars"
```

---

## Task 3: Adapter Factory + Auth — `gateway/run.py`

**Objective:** Wire ClawSwarm into the adapter factory and authorization maps.

**File:** Modify `gateway/run.py`

**Step 1:** Add to `_create_adapter()`:
```python
elif platform == Platform.CLAWSWARM:
    from gateway.platforms.clawswarm import ClawSwarmAdapter, check_clawswarm_requirements
    if not check_clawswarm_requirements():
        logger.warning("ClawSwarm: requirements not met — check env vars and websockets package")
        return None
    return ClawSwarmAdapter(config)
```

**Step 2:** Add to BOTH dicts in `_is_user_authorized()`:
```python
platform_env_map = {
    ...
    Platform.CLAWSWARM: "CLAWSWARM_ALLOWED_USERS",
}
platform_allow_all_map = {
    ...
    Platform.CLAWSWARM: "CLAWSWARM_ALLOW_ALL_USERS",
}
```

**Commit:**
```bash
git add gateway/run.py
git commit -m "feat(gateway): wire ClawSwarm into adapter factory and auth maps"
```

---

## Task 4: System Prompt Hint — `agent/prompt_builder.py`

**Objective:** Tell the agent it's running inside ClawSwarm group chat.

**File:** Modify `agent/prompt_builder.py`

Add to `PLATFORM_HINTS`:
```python
"clawswarm": (
    "You are operating inside a ClawSwarm multi-agent group chat room. "
    "You receive messages when @-mentioned. "
    "To involve other agents, use /discuss @name or /delegate @name on their own line. "
    "Plain text replies are shown to the group. Keep responses concise and actionable."
),
```

**Commit:**
```bash
git add agent/prompt_builder.py
git commit -m "feat(agent): add ClawSwarm platform hint to prompt builder"
```

---

## Task 5: Toolset — `toolsets.py`

**Objective:** Register a named toolset for ClawSwarm.

**File:** Modify `toolsets.py`

**Step 1:** Add toolset entry (after mattermost):
```python
"hermes-clawswarm": {
    "description": "ClawSwarm multi-agent group chat toolset",
    "tools": _HERMES_CORE_TOOLS,
    "includes": [],
},
```

**Step 2:** Add to `hermes-gateway` composite includes list:
```python
"hermes-gateway": {
    "includes": [..., "hermes-clawswarm"]
}
```

**Commit:**
```bash
git add toolsets.py
git commit -m "feat: add hermes-clawswarm toolset"
```

---

## Task 6: Cron Delivery — `cron/scheduler.py`

**Objective:** Allow cron jobs to deliver to ClawSwarm.

**File:** Modify `cron/scheduler.py`

Add to `platform_map` in `_deliver_result()`:
```python
"clawswarm": Platform.CLAWSWARM,
```

**Commit:**
```bash
git add cron/scheduler.py
git commit -m "feat(cron): add ClawSwarm to delivery platform map"
```

---

## Task 7: Send Message Tool — `tools/send_message_tool.py`

**Objective:** Allow `send_message` tool to target ClawSwarm rooms.

**File:** Modify `tools/send_message_tool.py`

**Step 1:** Add to `platform_map`:
```python
"clawswarm": Platform.CLAWSWARM,
```

**Step 2:** Add routing in `_send_to_platform()`:
```python
elif platform == Platform.CLAWSWARM:
    return await _send_clawswarm(pconfig, chat_id, message)
```

**Step 3:** Add standalone send function:
```python
async def _send_clawswarm(pconfig: PlatformConfig, chat_id: str, message: str) -> dict:
    """Send a single message to a ClawSwarm room (for cron/tool use)."""
    import websockets
    server_url = (pconfig.extra.get("server_url") or os.getenv("CLAWSWARM_SERVER_URL", "")).rstrip("/")
    token = pconfig.token or os.getenv("CLAWSWARM_TOKEN", "")
    agent_name = pconfig.extra.get("agent_name") or os.getenv("CLAWSWARM_AGENT_NAME", "hermes")
    room_id = chat_id.split(":")[0] if ":" in chat_id else chat_id

    ws_url = f"{server_url}/ws/agent"
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"type": "auth", "token": token, "agentName": agent_name}))
        await ws.send(json.dumps({"type": "message", "roomId": room_id, "content": message}))
    return {"success": True}
```

**Commit:**
```bash
git add tools/send_message_tool.py
git commit -m "feat(tools): add ClawSwarm support to send_message tool"
```

---

## Task 8: Status Display — `hermes_cli/status.py`

**Objective:** Show ClawSwarm connection status in `hermes status`.

**File:** Modify `hermes_cli/status.py`

Add to the `platforms` dict:
```python
"ClawSwarm": ("CLAWSWARM_TOKEN", "CLAWSWARM_HOME_CHANNEL"),
```

**Commit:**
```bash
git add hermes_cli/status.py
git commit -m "feat(cli): add ClawSwarm to status display"
```

---

## Task 9: Channel Directory — `gateway/channel_directory.py`

**Objective:** Enable session-based chat discovery for ClawSwarm.

**File:** Modify `gateway/channel_directory.py`

Add `"clawswarm"` to the session-based discovery loop:
```python
for plat_name in ("telegram", "whatsapp", "signal", "clawswarm", ...):
```

**Commit:**
```bash
git add gateway/channel_directory.py
git commit -m "feat(gateway): add ClawSwarm to channel directory"
```

---

## Task 10: Tests — `tests/gateway/test_clawswarm.py`

**Objective:** Basic test coverage for enum, config loading, adapter init, auth maps.

**File:** Create `tests/gateway/test_clawswarm.py`

```python
"""Tests for the ClawSwarm gateway adapter."""

import os
import pytest
from unittest.mock import patch

from gateway.config import Platform, _apply_env_overrides, GatewayConfig


def test_platform_enum_exists():
    assert Platform.CLAWSWARM.value == "clawswarm"


def test_config_loading_from_env():
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


def test_check_requirements_missing_url():
    from gateway.platforms.clawswarm import check_clawswarm_requirements
    with patch.dict(os.environ, {"CLAWSWARM_TOKEN": "ocs_x", "CLAWSWARM_AGENT_NAME": "h"}, clear=False):
        os.environ.pop("CLAWSWARM_SERVER_URL", None)
        assert check_clawswarm_requirements() is False


def test_check_requirements_ok(monkeypatch):
    from gateway.platforms.clawswarm import check_clawswarm_requirements
    monkeypatch.setenv("CLAWSWARM_SERVER_URL", "ws://localhost:3000")
    monkeypatch.setenv("CLAWSWARM_TOKEN", "ocs_x")
    monkeypatch.setenv("CLAWSWARM_AGENT_NAME", "hermes")
    assert check_clawswarm_requirements() is True


def test_authorization_maps():
    """ClawSwarm must appear in both auth maps in gateway/run.py."""
    import importlib, inspect
    run = importlib.import_module("gateway.run")
    src = inspect.getsource(run)
    assert "CLAWSWARM_ALLOWED_USERS" in src
    assert "CLAWSWARM_ALLOW_ALL_USERS" in src
```

**Commit:**
```bash
git add tests/gateway/test_clawswarm.py
git commit -m "test(gateway): add ClawSwarm adapter tests"
```

---

## Final Verification

```bash
cd ~/.hermes/hermes-agent
source venv/bin/activate

# Run new tests
python -m pytest tests/gateway/test_clawswarm.py -v

# Run full suite
python -m pytest tests/ -q

# Check all integration points covered
grep -r "telegram\|mattermost" gateway/ tools/ agent/ cron/ hermes_cli/ toolsets.py \
  --include="*.py" -l | sort -u
# For each file — verify clawswarm is also present
```

**配置方式（`.env` 或环境变量）：**
```bash
CLAWSWARM_SERVER_URL=ws://localhost:3000
CLAWSWARM_TOKEN=ocs_xxxxxxxxxxxxxxxxxxxx
CLAWSWARM_AGENT_NAME=hermes
CLAWSWARM_HOME_CHANNEL=your-room-id        # optional
CLAWSWARM_ALLOW_ALL_USERS=true             # or CLAWSWARM_ALLOWED_USERS=user1,user2
```
