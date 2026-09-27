import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.droid.runtime import DroidRuntime
from reachy_mini_conversation_app.conversation_handler import ConversationEvent


@pytest.mark.asyncio
async def test_runtime_emotes_after_silent_reply(tmp_path) -> None:
    """The runtime listens to the attached handler and queues the fallback emote."""
    runtime = DroidRuntime(MagicMock(), tmp_path)
    runtime._emotion_tool = AsyncMock(return_value={"status": "queued"})  # type: ignore[method-assign]
    handler = MagicMock()
    runtime.attach(handler)
    listener = handler.set_conversation_listener.call_args.args[0]

    for event in (
        ConversationEvent("user_transcript", "hello"),
        ConversationEvent("assistant_transcript", "Welcome back."),
        ConversationEvent("response_done"),
    ):
        listener(event)
    await asyncio.sleep(0)

    runtime._emotion_tool.assert_awaited_once_with(runtime.deps, emotion="happy")
