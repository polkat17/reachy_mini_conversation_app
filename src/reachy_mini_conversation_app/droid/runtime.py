"""The droid runtime: one object that listens to the conversation and drives droid behaviour."""

import asyncio
import logging
from typing import Any
from pathlib import Path
from collections.abc import Coroutine

from reachy_mini_conversation_app.droid.emotes import EmoteGuard
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion
from reachy_mini_conversation_app.conversation_handler import ConversationEvent, ConversationHandler


logger = logging.getLogger(__name__)


class DroidRuntime:
    """Owns the droid's behaviours and follows whichever conversation handler is active."""

    def __init__(self, deps: ToolDependencies, instance_path: str | Path | None) -> None:
        """Create the behaviours; nothing runs until a handler is attached."""
        self.deps = deps
        self.instance_path = instance_path
        self.handler: ConversationHandler | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._emotion_tool = PlayEmotion()
        self._emote_guard = EmoteGuard(self._queue_emotion)

    def attach(self, handler: ConversationHandler) -> None:
        """Listen to ``handler``; called for every handler the app builds."""
        self.handler = handler
        handler.set_conversation_listener(self._on_event)

    def _on_event(self, event: ConversationEvent) -> None:
        # Events are emitted from the handler's event loop, which is also where droid tasks run.
        self._loop = asyncio.get_running_loop()
        self._emote_guard.on_event(event)

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        if self._loop is None:
            coro.close()
            return
        task = self._loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _queue_emotion(self, emotion: str) -> None:
        logger.info("Droid fallback emote: %s", emotion)
        self._spawn(self._emotion_tool(self.deps, emotion=emotion))
