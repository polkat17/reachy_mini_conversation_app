"""The droid control page (``/droid``) and its JSON API on the app's settings server."""

from typing import Any
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from k3_droid.droid.runtime import DroidRuntime
from k3_droid.droid.updater import load_update_state
from k3_droid.tools.turn_body import turn_base


_PAGE = Path(__file__).parents[1] / "static" / "droid.html"


def droid_status(runtime: DroidRuntime) -> dict[str, Any]:
    """Return the state shown on the control page (no robot calls, so polling it costs nothing)."""
    state = load_update_state(runtime.data_dir)
    return {
        "droid_name": runtime.identity.droid_name,
        "owner_name": runtime.identity.owner_name,
        "memory": "on" if runtime.identity.memory_consent else "off",
        "movement": "running" if runtime.deps.movement_manager.is_running() else "stopped",
        "day_review": runtime.day_review_note,
        "update": (state.installed_revision[:8] or "unknown")
        + (f" ({state.last_result})" if state.last_result else ""),
    }


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

    @app.post("/droid/api/restore-motors")
    def _restore_motors() -> dict[str, Any]:
        return {"status": runtime.restore_motors()}

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
        return {"status": runtime.check_for_update()}
