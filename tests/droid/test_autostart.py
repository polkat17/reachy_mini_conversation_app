from typing import Any
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.droid import autostart as autostart_mod
from reachy_mini_conversation_app.daemon_api import DaemonApiError


def _fake_daemon(responses: dict[str, Any], calls: list[tuple[str, str, Any]]) -> Any:
    def request(_robot: Any, path: str, *, method: str = "GET", payload: Any = None, **_kwargs: Any) -> Any:
        calls.append((method, path, payload))
        return responses.get(f"{method} {path}", {})

    return request


def test_registers_running_app_as_startup_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app the daemon is running becomes the startup app."""
    calls: list[tuple[str, str, Any]] = []
    responses = {
        "GET /api/apps/current-app-status": {"info": {"name": "droid_app"}},
        "GET /api/apps/startup-app": {"startup_app": None},
    }
    monkeypatch.setattr(autostart_mod, "daemon_request", _fake_daemon(responses, calls))

    assert autostart_mod.ensure_autostart(MagicMock()) == "droid_app"
    assert ("PUT", "/api/apps/startup-app", {"startup_app": "droid_app"}) in calls


def test_already_registered_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """No PUT when the startup app is already this app."""
    calls: list[tuple[str, str, Any]] = []
    responses = {
        "GET /api/apps/current-app-status": {"info": {"name": "droid_app"}},
        "GET /api/apps/startup-app": {"startup_app": "droid_app"},
    }
    monkeypatch.setattr(autostart_mod, "daemon_request", _fake_daemon(responses, calls))

    assert autostart_mod.ensure_autostart(MagicMock()) == "droid_app"
    assert all(method == "GET" for method, _path, _payload in calls)


def test_not_run_by_daemon_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process started from the terminal has no current app and is not registered."""
    calls: list[tuple[str, str, Any]] = []
    monkeypatch.setattr(autostart_mod, "daemon_request", _fake_daemon({}, calls))

    assert autostart_mod.ensure_autostart(MagicMock()) is None


def test_daemon_errors_are_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """A daemon failure is logged, never raised into the app."""

    def failing(*_args: Any, **_kwargs: Any) -> Any:
        raise DaemonApiError("boom")

    monkeypatch.setattr(autostart_mod, "daemon_request", failing)

    assert autostart_mod.ensure_autostart(MagicMock()) is None
