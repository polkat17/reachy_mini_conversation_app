import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini_conversation_app.droid import runtime as runtime_mod
from reachy_mini_conversation_app.droid.web import register_droid_routes
from reachy_mini_conversation_app.droid.runtime import DroidRuntime
from reachy_mini_conversation_app.droid.presence import PresenceEvent
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


class _FakeStream:
    def __init__(self) -> None:
        self.is_dormant = False
        self.audio_tap = None
        self.asyncio_loop = None

    async def enter_dormant(self) -> None:
        self.is_dormant = True

    async def wake(self) -> None:
        self.is_dormant = False


def _runtime(tmp_path, monkeypatch: pytest.MonkeyPatch) -> DroidRuntime:
    monkeypatch.setattr(runtime_mod, "_GOODBYE_GRACE_S", 0.0)
    runtime = DroidRuntime(MagicMock(), tmp_path)
    runtime.bind_stream(_FakeStream())  # type: ignore[arg-type]
    runtime.beep = AsyncMock()  # type: ignore[method-assign]
    return runtime


@pytest.mark.asyncio
async def test_dormant_and_wake_cycle(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dormant parks the body and session; wake restores both and runs hooks."""
    runtime = _runtime(tmp_path, monkeypatch)
    dormant_reasons: list[str] = []
    wake_reasons: list[str] = []

    async def on_dormant(reason: str) -> None:
        dormant_reasons.append(reason)

    async def on_wake(reason: str) -> None:
        wake_reasons.append(reason)

    runtime.on_dormant.append(on_dormant)
    runtime.on_wake.append(on_wake)

    await runtime.enter_dormant("test")
    await runtime.enter_dormant("again")

    assert runtime.is_dormant
    assert dormant_reasons == ["test"]
    runtime.deps.reachy_mini.goto_sleep.assert_called_once()
    runtime.deps.movement_manager.stop.assert_called_once_with(reset_to_neutral=False)

    await runtime.wake("face")

    assert not runtime.is_dormant
    assert wake_reasons == ["face"]
    runtime.deps.reachy_mini.wake_up.assert_called_once()
    runtime.deps.movement_manager.start.assert_called_once()


@pytest.mark.asyncio
async def test_tick_goes_dormant_after_inactivity(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With nobody talking or in view, the droid goes dormant after the configured time."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime._loop = asyncio.get_running_loop()
    runtime.settings = dataclasses.replace(runtime.settings, dormant_after_minutes=1.0, idle_life=False)
    runtime._last_user_turn_at -= 120.0
    runtime.handler = MagicMock(last_activity_time=0.0)

    await runtime.tick()
    await asyncio.sleep(0.05)

    assert runtime.is_dormant


@pytest.mark.asyncio
async def test_loud_speech_wakes_only_without_wake_phrase(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Before a wake phrase exists, sustained speech wakes a dormant droid."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.request_wake = MagicMock()  # type: ignore[method-assign]
    runtime.stream.is_dormant = True  # type: ignore[union-attr]
    loud = np.full(16000, 0.3, dtype=np.float32)

    runtime._wake_on_loud_speech(loud)
    runtime.request_wake.assert_called_once_with("speech")

    runtime.identity.wake_phrase = "hey kay three"
    runtime._wake_on_loud_speech(loud)
    runtime.request_wake.assert_called_once()


def test_web_api_reports_status_and_wakes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The control page API exposes status and forwards wake requests."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.request_wake = MagicMock(return_value={"status": "waking"})  # type: ignore[method-assign]
    runtime.request_dormant = MagicMock(return_value={"error": "not running"})  # type: ignore[method-assign]
    app = FastAPI()
    register_droid_routes(app, runtime)
    client = TestClient(app)

    status = client.get("/droid/api/status").json()
    woke = client.post("/droid/api/wake")
    slept = client.post("/droid/api/sleep")

    assert status["droid_name"] == "K3-L0" and status["dormant"] is False
    assert woke.json() == {"status": "waking"}
    assert slept.status_code == 409
    assert client.get("/droid").status_code == 200


@pytest.mark.asyncio
async def test_owner_sighting_wakes_dormant_droid(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An activated, dormant droid wakes when it recognises its owner."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.identity.activated = True
    runtime.stream.is_dormant = True  # type: ignore[union-attr]
    runtime.wake = AsyncMock()  # type: ignore[method-assign]

    await runtime._react_to_presence(PresenceEvent("arrived", "Pasha", True, away_s=5000.0))

    runtime.wake.assert_awaited_once_with("owner_seen")


@pytest.mark.asyncio
async def test_stranger_is_greeted_with_startle(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An awake droid startles at a stranger and voices a guarded greeting."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.identity.activated = True
    runtime._emotion_tool = AsyncMock()  # type: ignore[method-assign]
    runtime.say_event = AsyncMock(return_value=True)  # type: ignore[method-assign]

    await runtime._react_to_presence(PresenceEvent("arrived", "stranger", False, away_s=float("inf")))

    runtime._emotion_tool.assert_awaited_once_with(runtime.deps, emotion="startled")
    assert "never share" in runtime.say_event.await_args.args[0]


@pytest.mark.asyncio
async def test_presence_ignored_before_activation(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """During activation, presence events do not interrupt the protocol."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.say_event = AsyncMock()  # type: ignore[method-assign]

    await runtime._react_to_presence(PresenceEvent("arrived", "stranger", False, away_s=float("inf")))

    runtime.say_event.assert_not_awaited()
