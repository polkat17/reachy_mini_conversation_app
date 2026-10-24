"""The droid control page (``/droid``) and its JSON API on the app's settings server."""

import os
import math
from typing import Any
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from k3_droid.daemon_api import DaemonApiError, daemon_request
from k3_droid.droid.notify import notifications_enabled
from k3_droid.app_lifecycle import robot_motor_state
from k3_droid.droid.runtime import DroidRuntime
from k3_droid.droid.updater import load_update_state
from k3_droid.droid.upgrades import upgrades_enabled
from k3_droid.tools.turn_body import turn_base


_PAGE = Path(__file__).parents[1] / "static" / "droid.html"


def droid_status(runtime: DroidRuntime) -> dict[str, Any]:
    """Return the state shown on the control page."""
    identity = runtime.identity
    runtime.day_review()  # refreshes day_review_note; the dataset is re-read at most every 10 minutes
    return {
        "droid_name": identity.droid_name,
        "owner_name": identity.owner_name,
        "activated": identity.activated,
        "dormant": runtime.is_dormant,
        "someone_present": runtime.someone_present(),
        "in_view": ", ".join(runtime.people_in_view()),
        "activation_stage": "" if identity.activated else runtime.activation.stage.id,
        "face_enrolled": runtime.face_registry.owner_enrolled(),
        "notifications": notifications_enabled(runtime.settings),
        "known_faces": [person.name for person in runtime.face_registry.people()],
        "mood": runtime.mood() if identity.activated else "",
        "update": _software_line(runtime),
        "upgrade_requests": upgrades_enabled(runtime.settings),
        "day_review": runtime.day_review_note,
        "movement": "running" if runtime.deps.movement_manager.is_running() else "stopped (asleep)",
        "connection": "ok" if runtime.robot_connected() else f"lost ({runtime.connection_note or 'just now'})",
        "load": _load_line(),
        "motors": _motors_line(runtime),
        "speaker": _speaker_line(runtime),
        "motor_watchdog": (
            f"{runtime.motor_restores} automatic restore(s); last: {runtime.motor_restore_note}"
            if runtime.motor_restores
            else "no restores needed"
        ),
        "movement_error": runtime.deps.movement_manager.last_command_error or "none",
        "base": _base_line(runtime),
    }


def _load_line() -> str:
    """Return the onboard computer's 1-minute load average against its core count."""
    try:
        load = os.getloadavg()[0]
    except OSError:
        return "unknown"
    cores = os.cpu_count() or 1
    return f"{load:.1f} on {cores} cores" + (" (overloaded)" if load > cores * 1.5 else "")


def _speaker_line(runtime: DroidRuntime) -> str:
    """Return the robot's speaker volume as the daemon reports it."""
    try:
        volume = daemon_request(runtime.deps.reachy_mini, "/api/volume/current").get("volume")
    except Exception as e:  # a status line must never break the page
        return f"unknown ({type(e).__name__})"
    return f"volume {volume}%" + (" (muted: raise it)" if volume == 0 else "")


def _motors_line(runtime: DroidRuntime) -> str:
    """Return the daemon's motor mode, and whether one of its own moves is blocking app targets."""
    mode, busy = robot_motor_state(runtime.deps.reachy_mini)
    if mode is None:
        return "unknown (no status from the robot)"
    return mode + (", robot busy with a built-in move (app commands ignored)" if busy else "")


def _base_line(runtime: DroidRuntime) -> str:
    """Commanded versus measured base angle, to tell a dropped command from a base that does not follow."""
    manager = runtime.deps.movement_manager
    try:
        commanded = manager.get_status()["last_commanded_pose"]["body_yaw"]
        head_joints, _ = runtime.deps.reachy_mini.get_current_joint_positions()
        measured = float(head_joints[0])
        tracking = "on" if manager.head_tracking_enabled() else "off"
    except Exception as e:
        return f"unavailable ({type(e).__name__})"
    commanded_text = "none yet" if commanded is None else f"{math.degrees(float(commanded)):.0f}°"
    return f"commanded {commanded_text}, measured {math.degrees(measured):.0f}°, face tracking {tracking}"


def _software_line(runtime: DroidRuntime) -> str:
    state = load_update_state(runtime.data_dir)
    if not runtime.settings.update_space:
        return "updates not set up"
    installed = state.installed_revision[:8] or "unknown"
    return f"{installed} from {runtime.settings.update_space}" + (
        f" ({state.last_result})" if state.last_result else ""
    )


def droid_memories(runtime: DroidRuntime) -> dict[str, Any]:
    """Return what the droid remembers, for the owner to review."""
    memory = runtime.memory
    return {
        "facts": [{"id": fact.id, "subject": fact.subject, "text": fact.text} for fact in memory.facts(limit=200)],
        "episodes": [
            {
                "id": episode.id,
                "when": f"{datetime.fromtimestamp(episode.ended_at):%Y-%m-%d %H:%M}",
                "text": episode.summary,
            }
            for episode in memory.episodes(limit=20)
        ],
        "reminders": [
            {
                "id": reminder.id,
                "when": f"{datetime.fromtimestamp(reminder.due_at):%Y-%m-%d %H:%M}",
                "text": reminder.text,
            }
            for reminder in memory.pending_reminders()
        ],
        "diary": [{"day": day, "text": memory.diary(day) or ""} for day in memory.diary_days(limit=7)],
    }


def _require_ok(result: dict[str, Any]) -> dict[str, Any]:
    if "error" in result:
        raise HTTPException(status_code=409, detail=result["error"])
    return result


def register_droid_routes(app: FastAPI, runtime: DroidRuntime) -> None:
    """Mount the control page and its API."""

    @app.get("/droid")
    def _page() -> FileResponse:
        return FileResponse(str(_PAGE))

    @app.get("/droid/api/status")
    def _status() -> dict[str, Any]:
        return droid_status(runtime)

    @app.post("/droid/api/wake")
    def _wake() -> dict[str, Any]:
        return _require_ok(runtime.request_wake("web"))

    @app.post("/droid/api/sleep")
    def _sleep() -> dict[str, Any]:
        return _require_ok(runtime.request_dormant("web"))

    @app.post("/droid/api/forget-face/{name}")
    def _forget_face(name: str) -> dict[str, Any]:
        if not runtime.face_registry.forget(name):
            raise HTTPException(status_code=404, detail=f"No face stored for {name}")
        return {"status": f"Forgot {name}'s face."}

    @app.post("/droid/api/factory-reset")
    def _factory_reset() -> dict[str, Any]:
        if not runtime.submit(runtime.factory_reset()):
            raise HTTPException(status_code=409, detail="the droid runtime is not running yet")
        return {"status": "Factory reset started. Activation will begin again."}

    @app.get("/droid/api/memories")
    def _memories() -> dict[str, Any]:
        return droid_memories(runtime)

    @app.post("/droid/api/forget-fact/{fact_id}")
    def _forget_fact(fact_id: int) -> dict[str, Any]:
        if not runtime.memory.delete_facts([fact_id]):
            raise HTTPException(status_code=404, detail="No such fact")
        return {"status": "Fact deleted."}

    @app.post("/droid/api/forget-episode/{episode_id}")
    def _forget_episode(episode_id: int) -> dict[str, Any]:
        if not runtime.memory.delete_episodes([episode_id]):
            raise HTTPException(status_code=404, detail="No such session")
        return {"status": "Session deleted."}

    @app.post("/droid/api/finish-activation")
    def _finish_activation() -> dict[str, Any]:
        if runtime.identity.activated:
            return {"status": "Already activated."}
        if not runtime.submit(runtime.complete_activation()):
            raise HTTPException(status_code=409, detail="the droid runtime is not running yet")
        return {"status": "Activated. Stages not done yet keep their defaults; say 'redo' to change one."}

    @app.post("/droid/api/restore-motors")
    def _restore_motors() -> dict[str, Any]:
        return {"status": runtime.restore_motors()}

    @app.post("/droid/api/test-sound")
    def _test_sound() -> dict[str, Any]:
        try:
            result = daemon_request(runtime.deps.reachy_mini, "/api/volume/test-sound", method="POST")
        except DaemonApiError as e:
            raise HTTPException(status_code=502, detail=f"The robot could not play the test sound: {e}") from e
        return {"status": f"Robot test sound: {result.get('message', 'played')}. Did you hear it?"}

    @app.post("/droid/api/volume/{level}")
    def _set_volume(level: int) -> dict[str, Any]:
        if not 0 <= level <= 100:
            raise HTTPException(status_code=400, detail="volume must be 0-100")
        try:
            daemon_request(runtime.deps.reachy_mini, "/api/volume/set", method="POST", payload={"volume": level})
        except DaemonApiError as e:
            raise HTTPException(status_code=502, detail=f"The robot did not change the volume: {e}") from e
        return {"status": f"Speaker volume set to {level}%."}

    @app.post("/droid/api/restart-app")
    def _restart_app() -> dict[str, Any]:
        return {"status": runtime.restart_app()}

    @app.post("/droid/api/turn/{direction}")
    def _turn(direction: str) -> dict[str, Any]:
        angles = {"left": 45.0, "right": -45.0, "home": None}
        if direction not in angles:
            raise HTTPException(status_code=404, detail="direction must be left, right or home")
        return _require_ok(turn_base(runtime.deps.movement_manager, angles[direction]))

    @app.post("/droid/api/update-now")
    def _update_now() -> dict[str, Any]:
        return {"status": runtime.check_for_update(force=True)}
