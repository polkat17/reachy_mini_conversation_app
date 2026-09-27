"""The droid control page (``/droid``) and its JSON API on the app's settings server."""

from typing import Any
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from reachy_mini_conversation_app.droid.notify import notifications_enabled
from reachy_mini_conversation_app.droid.runtime import DroidRuntime
from reachy_mini_conversation_app.droid.updater import load_update_state
from reachy_mini_conversation_app.droid.upgrades import upgrades_enabled


_PAGE = Path(__file__).parents[1] / "static" / "droid.html"


def droid_status(runtime: DroidRuntime) -> dict[str, Any]:
    """Return the state shown on the control page."""
    identity = runtime.identity
    return {
        "droid_name": identity.droid_name,
        "owner_name": identity.owner_name,
        "activated": identity.activated,
        "dormant": runtime.is_dormant,
        "wake_phrase": identity.wake_phrase,
        "someone_present": runtime.someone_present(),
        "in_view": ", ".join(runtime.people_in_view()),
        "activation_stage": "" if identity.activated else runtime.activation.stage.id,
        "face_enrolled": runtime.face_registry.owner_enrolled(),
        "notifications": notifications_enabled(runtime.settings),
        "known_faces": [person.name for person in runtime.face_registry.people()],
        "mood": runtime.mood() if identity.activated else "",
        "update": _software_line(runtime),
        "upgrade_requests": upgrades_enabled(runtime.settings),
    }


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

    @app.post("/droid/api/update-now")
    def _update_now() -> dict[str, Any]:
        return {"status": runtime.check_for_update(force=True)}
