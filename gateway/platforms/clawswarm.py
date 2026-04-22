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

    async def connect(self) -> bool:
        self._closing = False
        self._ws_task = asyncio.create_task(self._ws_loop())
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
        try:
            msg = json.loads(raw)
        except Exception:
            return

        if msg.get("type") != "message":
            return
        if not msg.get("_context"):
            return

        self._room_id = msg.get("roomId")
        context = msg["_context"]

        content: str = msg.get("content", "")
        from_user: str = msg.get("from", "unknown")
        msg_id: str = msg.get("id", "")
        room_id: str = msg.get("roomId", "")

        delegate_m = re.search(r'(?:^|\n)/delegate\s+@\S+\s+([\s\S]+)', content, re.IGNORECASE)
        discuss_m = re.search(r'(?:^|\n)/discuss\s+@\S+\s+([\s\S]+)', content, re.IGNORECASE)
        effective_body = (delegate_m or discuss_m)
        if effective_body:
            effective_body = effective_body.group(1).strip()
        else:
            effective_body = content

        chat_id = f"{room_id}:{msg_id}"

        context_lines: list[str] = []

        agent_name = context.get("agentName") or self._agent_name
        agent_nickname = context.get("agentNickname", "")
        self_display = f"{agent_nickname} (@{agent_name})" if agent_nickname else f"@{agent_name}"
        context_lines.append(
            f"YOUR IDENTITY: You are {self_display}. This is your name in this group chat."
        )

        task_label = context.get("taskLabel", "")
        context_lines.append(
            f"Task scope: {repr(task_label) if task_label else '(no label)'}. "
            "Only use the conversation history provided as context."
        )

        if context.get("agentRole"):
            context_lines.append(f"Your role: {context['agentRole']}")

        other_agents = context.get("roomAgents", [])
        if other_agents:
            agent_list = []
            for a in other_agents:
                display = f"{a['nickname']} (@{a['name']})" if a.get("nickname") else f"@{a['name']}"
                agent_list.append("- " + display + (f" — {a['role']}" if a.get("role") else ""))
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

    async def send(self, chat_id: str, text: str, **kwargs) -> SendResult:
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
