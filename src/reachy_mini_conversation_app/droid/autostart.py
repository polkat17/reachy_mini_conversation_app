"""Register this app as the daemon's startup app so the droid starts on every power-on."""

import logging
from typing import Any

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.daemon_api import DaemonApiError, daemon_request


logger = logging.getLogger(__name__)

_CURRENT_APP_PATH = "/api/apps/current-app-status"
_STARTUP_APP_PATH = "/api/apps/startup-app"


def current_app_name(robot: ReachyMini) -> str | None:
    """Return this app's name as the daemon knows it, or None when it was not started by the daemon."""
    try:
        status: Any = daemon_request(robot, _CURRENT_APP_PATH)
    except DaemonApiError as e:
        logger.warning("Could not read the current app from the daemon: %s", e)
        return None
    app_name = status.get("info", {}).get("name") if isinstance(status, dict) else None
    if not isinstance(app_name, str) or not app_name:
        logger.info("This process was not started by the daemon's app manager")
        return None
    return app_name


def ensure_autostart(robot: ReachyMini, app_name: str) -> bool:
    """Make ``app_name`` the daemon's startup app; returns whether it is registered."""
    try:
        current: Any = daemon_request(robot, _STARTUP_APP_PATH)
        if isinstance(current, dict) and current.get("startup_app") == app_name:
            return True
        daemon_request(robot, _STARTUP_APP_PATH, method="PUT", payload={"startup_app": app_name})
    except DaemonApiError as e:
        logger.warning("Failed to register the droid as the startup app: %s", e)
        return False
    logger.info("Registered %s as the startup app; it will start on every power-on", app_name)
    return True
