import dataclasses
from typing import Any

import httpx
import pytest

from reachy_mini_conversation_app.droid import notify as notify_mod
from reachy_mini_conversation_app.droid.settings import DroidSettings


def _settings(topic: str) -> DroidSettings:
    return dataclasses.replace(DroidSettings.from_env(), ntfy_topic=topic, ntfy_server="https://ntfy.example")


@pytest.mark.asyncio
async def test_notification_skipped_without_topic() -> None:
    """No topic means no network call."""
    assert await notify_mod.send_notification(_settings(""), "hello") is False


@pytest.mark.asyncio
async def test_notification_posts_to_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message is POSTed to the topic URL with ASCII-safe headers."""
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    real_client = httpx.AsyncClient

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(notify_mod.httpx, "AsyncClient", client_factory)

    sent = await notify_mod.send_notification(_settings("droid-123"), "Привет", title="Hi ✨", priority="high")

    assert sent is True
    assert str(requests[0].url) == "https://ntfy.example/droid-123"
    assert requests[0].headers["Title"] == "Hi "
    assert requests[0].content.decode("utf-8") == "Привет"


@pytest.mark.asyncio
async def test_notification_failure_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP errors are logged and reported as not sent."""
    real_client = httpx.AsyncClient

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(lambda _r: httpx.Response(500)), **kwargs)

    monkeypatch.setattr(notify_mod.httpx, "AsyncClient", client_factory)

    assert await notify_mod.send_notification(_settings("t"), "x") is False
