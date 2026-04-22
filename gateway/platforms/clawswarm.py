"""ClawSwarm gateway adapter.

Connects Hermes to a ClawSwarm multi-agent group chat server via WebSocket.
Hermes acts as a WS client; it authenticates, then responds when @-mentioned.

Environment variables:
    CLAWSWARM_SERVER_URL       WebSocket base URL (e.g. ws://localhost:3000)
    CLAWSWARM_TOKEN            Auth token (starts with ocs_)
    CLAWSWARM_AGENT_NAME       Display name for this Hermes instance
    CLAWSWARM_ALLOWED_USERS    Comma-separated allowed user names (optional)
    CLAWSWARM_HOME_CHANNEL     Room ID for cron/notification delivery (optional)

Protocol reference: channel.js (ClawSwarm OpenClaw plugin)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
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

# Maximum content length to process — protects against pathological payloads
_MAX_CONTENT_LEN = 64_000
# Maximum history entries to include
_MAX_HISTORY_ENTRIES = 100


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

        self._ws: Any = None
        self._ws_task: Optional[asyncio.Task] = None
        self._closing = False
        self._room_id: Optional[str] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        self._closing = False
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._mark_connected()
        logger.info("ClawSwarm: connecting to %s as %s", self._server_url, self._agent_name)
        return True

    async def disconnect(self) -> None:
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
        import websockets

        delay = _RECONNECT_BASE_DELAY
        ws_url = f"{self._server_url}/ws/agent"

        while not self._closing:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws = ws
                    delay = _RECONNECT_BASE_DELAY

                    await ws.send(json.dumps({
                        "type": "auth",
                        "token": self._token,
                        "agentName": self._agent_name,
                    }))
                    logger.info("ClawSwarm: authenticated as %s", self._agent_name)

                    async for raw in ws:
                        if self._closing:
                            break
                        try:
                            await self._handle_raw(raw)
                        except Exception as exc:
                            logger.exception("ClawSwarm: error handling message: %s", exc)

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

    # ── Inbound message handler ────────────────────────────────────────────

    async def _handle_raw(self, raw: str | bytes) -> None:
        """Parse and dispatch an inbound WS frame.

        Mirrors channel.js handleInbound() logic:
        - No _context  → plain message, dispatch content as-is
        - Has _context → enrich with role, skills, agents list, history
        """
        try:
            msg = json.loads(raw)
        except Exception:
            logger.debug("ClawSwarm: failed to parse JSON frame")
            return

        if not isinstance(msg, dict):
            return

        msg_type = msg.get("type", "")

        # Diagnostic logging for server control frames
        if msg_type in ("auth_ok", "auth_fail", "status", "error"):
            logger.info("ClawSwarm server: %s", msg)
            # auth_fail is fatal — stop reconnecting to avoid token-spam loop
            if msg_type == "auth_fail":
                logger.error("ClawSwarm: authentication failed — stopping reconnect loop")
                self._closing = True
            return

        if msg_type != "message":
            return

        room_id: str = msg.get("roomId") or ""
        msg_id: str = msg.get("id") or ""
        from_user: str = msg.get("from") or "unknown"
        content: str = msg.get("content") or ""

        # Guard: must have roomId to be routable
        if not room_id:
            logger.warning("ClawSwarm: received message without roomId — dropping")
            return

        # Guard: skip empty content (keep-alive pings, etc.)
        if not content.strip():
            return

        # Guard: truncate pathological payloads
        if len(content) > _MAX_CONTENT_LEN:
            logger.warning(
                "ClawSwarm: content too long (%d chars) — truncating to %d",
                len(content), _MAX_CONTENT_LEN,
            )
            content = content[:_MAX_CONTENT_LEN]

        # Update last-known room for fallback delivery
        self._room_id = room_id

        # _context is present for @-mentioned messages; absent for direct/admin messages
        context = msg.get("_context")
        if context is not None and not isinstance(context, dict):
            logger.warning("ClawSwarm: _context is not a dict (%r) — ignoring context", type(context))
            context = None

        # Only process messages that have _context (i.e. this agent was @-mentioned
        # or targeted by /delegate or /discuss). Without this guard, the same message
        # arrives TWICE: once via broadcastToRoom (no _context) and once via the
        # direct per-agent send (with _context). The first delivery would start a
        # session, and the second would see it as busy and emit "Interrupting current task".
        if not context:
            return

        # Strip /delegate @name or /discuss @name prefix — keep the actual task body
        delegate_m = re.search(r'(?:^|\n)/delegate\s+@\S+\s+([\s\S]+)', content, re.IGNORECASE)
        discuss_m = re.search(r'(?:^|\n)/discuss\s+@\S+\s+([\s\S]+)', content, re.IGNORECASE)
        match = delegate_m or discuss_m
        effective_body: str = match.group(1).strip() if match else content

        # session key: one session per (room, message) — clean slate per mention,
        # matching Hermes's stateless-per-task model (channel.js uses per-room sessions,
        # but Hermes manages its own history via SessionStore).
        chat_id = f"{room_id}:{msg_id}"

        # Build system context lines from _context (channel.js buildBody equivalent)
        context_lines: list[str] = []

        if context:
            # Identity
            agent_name = context.get("agentName") or self._agent_name
            agent_nickname = context.get("agentNickname") or ""
            self_display = (
                f"{agent_nickname} (@{agent_name})" if agent_nickname else f"@{agent_name}"
            )
            context_lines.append(
                f"YOUR IDENTITY: You are {self_display}. This is your name in this group chat."
            )

            # Task label
            task_label = context.get("taskLabel") or ""
            context_lines.append(
                f"Task scope: {repr(task_label) if task_label else '(no label)'}. "
                "Only use the conversation history provided as context."
            )

            # Role (mirrors channel.js [System Context] block)
            agent_role = context.get("agentRole") or ""
            if agent_role:
                context_lines.append(f"Your role: {agent_role}")

            # Skills (channel.js: ctx.agentSkills — was missing in previous version)
            agent_skills = context.get("agentSkills")
            if isinstance(agent_skills, list) and agent_skills:
                context_lines.append(f"Your skills: {', '.join(str(s) for s in agent_skills)}")

            # Other agents in the room
            other_agents = context.get("roomAgents")
            if isinstance(other_agents, list) and other_agents:
                agent_list = []
                for a in other_agents:
                    if not isinstance(a, dict):
                        continue
                    display = (
                        f"{a['nickname']} (@{a['name']})"
                        if a.get("nickname") else f"@{a['name']}"
                    )
                    role_suffix = f" — {a['role']}" if a.get("role") else ""
                    agent_list.append(f"- {display}{role_suffix}")
                if agent_list:
                    context_lines.append(
                        "Other agents in this room:\n" + "\n".join(agent_list) + "\n\n"
                        "HOW TO INTERACT WITH OTHER AGENTS:\n"
                        "  /discuss @name <content> — ask a question or continue discussion\n"
                        "  /delegate @name <task>   — formally hand off a task\n"
                        "Commands must appear on their own line. Only ONE command per message."
                    )

            # Task history (mirrors channel.js taskHistory block)
            # channel.js format: "from (fromType): content"
            task_history = context.get("taskHistory")
            if isinstance(task_history, list) and task_history:
                history_lines = ["## Conversation history (most recent last)"]
                for h in task_history[-_MAX_HISTORY_ENTRIES:]:
                    if not isinstance(h, dict):
                        continue
                    if h.get("id") == msg_id:
                        # Skip the current message itself (already in effective_body)
                        continue
                    h_from = h.get("from") or "unknown"
                    h_type = h.get("fromType") or "human"
                    h_content = h.get("content") or ""
                    history_lines.append(f"{h_from} ({h_type}): {h_content}")
                if len(history_lines) > 1:
                    context_lines.append("\n".join(history_lines))

        extra_system = "\n\n".join(context_lines)

        source = self.build_source(
            chat_id=chat_id,
            chat_type="group",
            user_id=from_user,
            user_name=from_user,
        )

        event = MessageEvent(
            message_id=msg_id,
            text=effective_body,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=msg,
        )
        event.extra = {"extra_system": extra_system, "room_id": room_id}

        await self.handle_message(event)

    # ── Outbound ───────────────────────────────────────────────────────────

    async def send(self, chat_id: str, content: str = "", **kwargs) -> SendResult:
        """Send a reply to the room.

        chat_id format: "{room_id}:{msg_id}" or bare "{room_id}".
        """
        if ":" in chat_id:
            room_id = chat_id.split(":", 1)[0]
        else:
            room_id = chat_id or kwargs.get("room_id") or self._room_id

        # Accept legacy 'text' kwarg
        text = content or kwargs.get("text", "")

        if not room_id:
            return SendResult(success=False, error="room_id unknown")

        ws = self._ws
        if not ws:
            return SendResult(success=False, error="not connected")

        # Guard: check websocket is still open (mirrors channel.js readyState check)
        try:
            if not ws.open:
                return SendResult(success=False, error="websocket closed")
        except AttributeError:
            pass  # older websockets versions — fall through and let send() raise

        try:
            await ws.send(json.dumps({
                "type": "message",
                "roomId": room_id,
                "content": text,
            }))
            return SendResult(success=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, **kwargs) -> None:
        if ":" in chat_id:
            room_id = chat_id.split(":", 1)[0]
        else:
            room_id = chat_id or kwargs.get("room_id") or self._room_id

        ws = self._ws
        if not room_id or not ws:
            return

        try:
            if not ws.open:
                return
        except AttributeError:
            pass

        try:
            await ws.send(json.dumps({
                "type": "typing",
                "roomId": room_id,
                "status": "start",
            }))
        except Exception:
            pass

    async def stop_typing(self, chat_id: str) -> None:
        if ":" in chat_id:
            room_id = chat_id.split(":", 1)[0]
        else:
            room_id = chat_id or self._room_id

        ws = self._ws
        if not room_id or not ws:
            return

        try:
            if not ws.open:
                return
        except AttributeError:
            pass

        try:
            await ws.send(json.dumps({
                "type": "typing",
                "roomId": room_id,
                "status": "stop",
            }))
        except Exception:
            pass

    async def get_chat_info(self, chat_id: str) -> dict:
        room_id = chat_id.split(":")[0] if ":" in chat_id else chat_id
        return {"name": f"ClawSwarm room {room_id}", "type": "group", "chat_id": chat_id}
