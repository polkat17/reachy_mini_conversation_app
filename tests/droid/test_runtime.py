import time
import asyncio
import dataclasses
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini_conversation_app.droid import runtime as runtime_mod
from reachy_mini_conversation_app.droid.web import register_droid_routes
from reachy_mini_conversation_app.droid.senses import SoundEvent
from reachy_mini_conversation_app.droid.journal import SessionSummary
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
        self.output_filter = None
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


@pytest.mark.asyncio
async def test_checkpoint_saves_episode_and_facts(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Going dormant turns the transcript into an episode plus durable facts."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.identity.activated = True
    runtime.identity.memory_consent = True
    monkeypatch.setattr(
        runtime_mod,
        "summarize_session",
        lambda *_args: SessionSummary(
            "Talked about Anna's visit.", "happy", ["family"], [("Anna", "Anna is Pasha's sister")]
        ),
    )
    runtime._record_transcript(ConversationEvent("user_transcript", "Anna visits Friday"))
    runtime._record_transcript(ConversationEvent("assistant_transcript", "Noted."))

    await runtime.enter_dormant("test")

    assert runtime.memory.episodes()[0].summary == "Talked about Anna's visit."
    assert runtime.memory.facts()[0].text == "Anna is Pasha's sister"
    assert runtime.journal.lines == []


@pytest.mark.asyncio
async def test_due_reminder_is_spoken_and_sent_when_owner_away(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Due reminders are voiced, marked delivered, and pushed to the phone when the owner is not in view."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.say_event = AsyncMock(return_value=True)  # type: ignore[method-assign]
    runtime.handler = MagicMock(_is_connected=MagicMock(return_value=True))
    notify = AsyncMock(return_value=True)
    monkeypatch.setattr(runtime_mod, "send_notification", notify)
    runtime.memory.add_reminder(0.0, "Take the pizza out")

    await runtime._deliver_due_reminders()
    await runtime._deliver_due_reminders()

    runtime.say_event.assert_awaited_once()
    assert "Take the pizza out" in runtime.say_event.await_args.args[0]
    notify.assert_awaited_once()


@pytest.mark.asyncio
async def test_due_reminder_wakes_dormant_droid(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reminder falling due while dormant wakes the droid."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.stream.is_dormant = True  # type: ignore[union-attr]
    runtime.wake = AsyncMock()  # type: ignore[method-assign]
    runtime.memory.add_reminder(0.0, "Stand-up meeting")

    await runtime.tick()

    runtime.wake.assert_awaited_once_with("reminder")


@pytest.mark.asyncio
async def test_due_reminder_waits_for_session(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A just-due reminder waits while the session is still connecting."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.say_event = AsyncMock(return_value=True)  # type: ignore[method-assign]
    runtime.handler = MagicMock(_is_connected=MagicMock(return_value=False))
    runtime.memory.add_reminder(time.time() - 5, "Stretch")

    await runtime._deliver_due_reminders()

    runtime.say_event.assert_not_awaited()
    assert len(runtime.memory.pending_reminders()) == 1


@pytest.mark.asyncio
async def test_music_starts_a_dance_and_silence_stops_it(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Music makes an idle, activated droid dance; when it stops, the dance stops."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.identity.activated = True
    runtime.handler = MagicMock(last_activity_time=0.0)
    runtime._dance_tool = AsyncMock()  # type: ignore[method-assign]
    runtime._stop_dance_tool = AsyncMock()  # type: ignore[method-assign]

    await runtime._react_to_sound(SoundEvent("music_started", 0.6))
    await runtime._react_to_sound(SoundEvent("music_stopped", 0.0))

    runtime._dance_tool.assert_awaited_once()
    runtime._stop_dance_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_cat_reaction_has_cooldown_and_counts_visits(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cat gets one greeting per cooldown, and visits are counted for the diary."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.identity.activated = True
    runtime.say_event = AsyncMock(return_value=True)  # type: ignore[method-assign]

    await runtime._react_to_sound(SoundEvent("meow", 0.8))
    await runtime._react_to_sound(SoundEvent("meow", 0.8))

    runtime.say_event.assert_awaited_once()
    assert "meowing" in runtime.say_event.await_args.args[0]
    assert runtime.memory.get_state(f"cat_visits:{datetime.now():%Y-%m-%d}") == "1"


def test_cat_is_not_a_person(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cat in view does not keep the droid awake or lock memories away."""
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.presence = MagicMock()
    runtime.presence.tracker.present.return_value = ["cat"]

    assert runtime.people_in_view() == [] and runtime.someone_present() is False
    assert runtime.memory_access_allowed() == (True, "")


def test_voice_fx_is_installed_on_the_stream(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With voice FX on, the stream's output filter is the droid filter."""
    runtime = _runtime(tmp_path, monkeypatch)
    chunk = np.full(800, 0.1, dtype=np.float32)

    filtered = runtime.stream.output_filter(16000, chunk)  # type: ignore[union-attr]

    assert filtered.shape == chunk.shape and not np.allclose(filtered, chunk)
