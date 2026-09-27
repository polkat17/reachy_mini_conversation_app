from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.tools import notify_owner as notify_owner_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_notify_owner_requires_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without DROID_NTFY_TOPIC the tool explains notifications are not set up."""
    monkeypatch.delenv("DROID_NTFY_TOPIC", raising=False)

    result = await notify_owner_mod.NotifyOwner()(_deps(), message="hi")

    assert "not set up" in result["error"]


@pytest.mark.asyncio
async def test_notify_owner_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a topic the message goes out with the requested priority."""
    monkeypatch.setenv("DROID_NTFY_TOPIC", "droid-topic")
    send = AsyncMock(return_value=True)
    monkeypatch.setattr(notify_owner_mod, "send_notification", send)

    result = await notify_owner_mod.NotifyOwner()(_deps(), message="Door is open", urgent=True)

    assert result == {"status": "sent"}
    assert send.await_args.kwargs["priority"] == "high"
