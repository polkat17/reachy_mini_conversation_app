import dataclasses
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from k3_droid.droid import runtime as runtime_mod
from k3_droid.droid.web import register_droid_routes
from k3_droid.droid.journal import SessionSummary
from k3_droid.droid.runtime import DroidRuntime
from k3_droid.droid.updater import HEALTH_FILENAME, UpdateState, load_update_state, save_update_state
from k3_droid.conversation_handler import ConversationEvent


def _runtime(tmp_path) -> DroidRuntime:  # noqa: ANN001
    return DroidRuntime(MagicMock(), tmp_path)


def _listener(runtime: DroidRuntime):  # noqa: ANN202
    handler = MagicMock()
    runtime.attach(handler)
    return handler.set_conversation_listener.call_args.args[0]


def test_runtime_records_transcripts_only(tmp_path) -> None:  # noqa: ANN001
    """Final user and assistant lines go into the journal; other events are ignored."""
    runtime = _runtime(tmp_path)
    listener = _listener(runtime)

    for event in (
        ConversationEvent("session_started"),
        ConversationEvent("user_transcript", "Привет"),
        ConversationEvent("tool_call", "play_emotion"),
        ConversationEvent("assistant_transcript", "Здравствуй."),
        ConversationEvent("response_done"),
    ):
        listener(event)

    assert runtime.journal.lines == [("user", "Привет"), ("assistant", "Здравствуй.")]


def test_save_session_stores_episode_and_facts(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """Saving turns the transcript into an episode plus durable facts and starts a new transcript."""
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(
        runtime_mod,
        "summarize_session",
        lambda *_args: SessionSummary(
            "Talked about Anna's visit.", "happy", ["family"], [("Anna", "Anna is Pasha's sister")]
        ),
    )
    runtime.journal.add("user", "Anna visits Friday")
    runtime.journal.add("assistant", "Noted.")

    runtime.save_session("test")

    assert runtime.memory.episodes()[0].summary == "Talked about Anna's visit."
    assert runtime.memory.facts()[0].text == "Anna is Pasha's sister"
    assert runtime.journal.lines == []


def test_save_session_respects_memory_consent(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """With memory switched off nothing is summarised."""
    runtime = _runtime(tmp_path)
    runtime.identity.memory_consent = False
    summarize = MagicMock()
    monkeypatch.setattr(runtime_mod, "summarize_session", summarize)
    runtime.journal.add("user", "secret")
    runtime.journal.add("assistant", "ok")

    runtime.save_session("test")

    summarize.assert_not_called()
    assert runtime.memory.episodes() == []


def test_a_failed_summary_never_raises(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """The summariser runs in the background; an LLM error is logged, not raised."""
    runtime = _runtime(tmp_path)

    def boom(*_args: object) -> SessionSummary:
        raise RuntimeError("LLM down")

    monkeypatch.setattr(runtime_mod, "summarize_session", boom)
    runtime.journal.add("user", "hi")
    runtime.journal.add("assistant", "hello")

    runtime.save_session("test")

    assert runtime.memory.episodes() == []


def test_start_writes_the_update_health_beacon(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """An update waits for this beacon from the new version; without it, it rolls back."""
    monkeypatch.setenv("DROID_AUTOSTART", "0")
    runtime = DroidRuntime(MagicMock(), tmp_path)
    runtime.start()
    try:
        assert (runtime.data_dir / HEALTH_FILENAME).exists()
    finally:
        runtime._stop.set()


def _update_runtime(tmp_path) -> DroidRuntime:  # noqa: ANN001
    runtime = _runtime(tmp_path)
    runtime.settings = dataclasses.replace(runtime.settings, update_space="owner/droid")
    runtime.app_name = "droid_app"
    runtime.deps.reachy_mini.client.host, runtime.deps.reachy_mini.client.port = "localhost", 8000
    return runtime


def test_update_check_launches_helper_for_new_revision(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """A newer Space revision starts the detached helper with the previous revision for rollback."""
    runtime = _update_runtime(tmp_path)
    save_update_state(runtime.data_dir, UpdateState(installed_revision="old-sha"))
    monkeypatch.setattr(runtime_mod, "latest_revision", lambda *_args: "new-sha")
    monkeypatch.setattr(runtime_mod, "installed_revision", lambda *_args: None)
    launch = MagicMock()
    monkeypatch.setattr(runtime_mod, "launch_update_helper", launch)

    result = runtime.check_for_update()

    args = launch.call_args.args[0]
    assert result.startswith("Installing") and args["revision"] == "new-sha" and args["previous"] == "old-sha"
    assert args["daemon_url"] == "http://localhost:8000" and args["boot_dormant"] == "0"


def test_update_check_reports_up_to_date_from_the_installed_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,  # noqa: ANN001
) -> None:
    """When pip's record matches the Space, nothing is installed."""
    runtime = _update_runtime(tmp_path)
    monkeypatch.setattr(runtime_mod, "latest_revision", lambda *_args: "sha-2")
    monkeypatch.setattr(runtime_mod, "installed_revision", lambda *_args: "sha-2")
    launch = MagicMock()
    monkeypatch.setattr(runtime_mod, "launch_update_helper", launch)

    assert runtime.check_for_update() == "Already up to date."
    assert load_update_state(runtime.data_dir).installed_revision == "sha-2"
    launch.assert_not_called()


def test_restore_motors_clears_stuck_moves_and_wakes_the_body(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """Restore motors stops daemon moves still listed as running, then turns torque on and restarts movement."""
    runtime = _runtime(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_daemon(robot, path, method="GET", payload=None):  # noqa: ANN001, ANN202
        calls.append((method, path))
        return [{"uuid": "stuck"}] if path == "/api/move/running" else {}

    monkeypatch.setattr(runtime_mod, "daemon_request", fake_daemon)
    manager = runtime.deps.movement_manager

    result = runtime.restore_motors()

    assert result.startswith("Motors restored")
    assert ("POST", "/api/move/stop") in calls
    runtime.deps.reachy_mini.enable_motors.assert_called_once_with()
    runtime.deps.reachy_mini.wake_up.assert_not_called()  # its sound can leave an open session silent
    manager.stop.assert_called_once_with(reset_to_neutral=False)
    manager.start.assert_called_once_with()


def test_restore_motors_restarts_movement_even_when_a_robot_call_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,  # noqa: ANN001
) -> None:
    """A dropped link mid-restore must not leave the control loop stopped."""
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(runtime_mod, "daemon_request", lambda *args, **kwargs: [])
    runtime.deps.reachy_mini.enable_wobbling.side_effect = ConnectionError("Lost connection with the server.")

    result = runtime.restore_motors()

    assert result.startswith("Could not switch the motors on")
    runtime.deps.movement_manager.start.assert_called_once_with()


def test_control_page_reports_status_memories_and_turns(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """The control page shows status and memories and turns the base without the conversation."""
    runtime = _runtime(tmp_path)
    runtime.memory.remember("Pasha likes tea", "Pasha")
    manager = runtime.deps.movement_manager
    manager.is_running.return_value = True
    manager.body_heading_target.return_value = 0.0
    manager.turn_body.side_effect = lambda heading, duration: heading
    app = FastAPI()
    register_droid_routes(app, runtime)
    client = TestClient(app)

    status = client.get("/droid/api/status").json()
    memories = client.get("/droid/api/memories").json()
    turned = client.post("/droid/api/turn/left").json()

    assert status["droid_name"] == "K3-L0" and status["movement"] == "running" and status["memory"] == "on"
    assert memories["facts"][0]["text"] == "Pasha likes tea"
    assert turned["status"] == "turning to 45 degrees from the starting direction"
    assert client.post("/droid/api/turn/up").status_code == 404


def test_reply_plays_a_fitting_emotion_and_a_request_wins(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """The droid's own words pick its body language; an explicit request plays exactly what was asked."""
    runtime = _runtime(tmp_path)
    played: list[str] = []
    monkeypatch.setattr(runtime, "_play_emotion", played.append)

    class _InlineThread:
        def __init__(self, target, args, **_kwargs) -> None:  # noqa: ANN001
            self._run = lambda: target(*args)

        def start(self) -> None:
            self._run()

    monkeypatch.setattr(runtime_mod.threading, "Thread", _InlineThread)
    listener = _listener(runtime)

    listener(ConversationEvent("user_transcript", "Как дела?"))
    listener(ConversationEvent("assistant_transcript", "Сейчас пятнадцать градусов."))
    listener(ConversationEvent("user_transcript", "Покажи грусть"))
    listener(ConversationEvent("assistant_transcript", "Вот так."))
    listener(ConversationEvent("user_transcript", "Я сдал экзамен!"))
    listener(ConversationEvent("assistant_transcript", "Поздравляю!"))  # too soon after the last emotion

    assert played == ["sad"]
