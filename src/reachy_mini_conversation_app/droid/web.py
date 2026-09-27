"""The droid control page (``/droid``) and its JSON API on the app's settings server."""

from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from reachy_mini_conversation_app.droid.notify import notifications_enabled
from reachy_mini_conversation_app.droid.runtime import DroidRuntime


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
        "notifications": notifications_enabled(runtime.settings),
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
