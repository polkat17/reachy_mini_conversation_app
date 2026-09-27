"""Register this app as the daemon's startup app so the droid starts on every power-on."""

import logging
from typing import Any

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.daemon_api import DaemonApiError, daemon_request


logger = logging.getLogger(__name__)

_CURRENT_APP_PATH = "/api/apps/current-app-status"
_STARTUP_APP_PATH = "/api/apps/startup-app"


def ensure_autostart(robot: ReachyMini) -> str | None:
    """Make the currently running app the startup app; returns its name, or None when not run by the daemon."""
    try:
        status: Any = daemon_request(robot, _CURRENT_APP_PATH)
        app_name = status.get("info", {}).get("name") if isinstance(status, dict) else None
        if not isinstance(app_name, str) or not app_name:
            logger.info("Autostart skipped: this process was not started by the daemon's app manager")
            return None

        current: Any = daemon_request(robot, _STARTUP_APP_PATH)
        if isinstance(current, dict) and current.get("startup_app") == app_name:
            return app_name

        daemon_request(robot, _STARTUP_APP_PATH, method="PUT", payload={"startup_app": app_name})
    except DaemonApiError as e:
        logger.warning("Failed to register the droid as the startup app: %s", e)
        return None
    logger.info("Registered %s as the startup app; it will start on every power-on", app_name)
    return app_name
