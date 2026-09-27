"""Droid settings read from the environment (documented in ``.env.example``)."""

import os
import logging
from dataclasses import dataclass

from reachy_mini_conversation_app.config import _env_flag


logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s.", name, raw, default)
        return default


@dataclass(frozen=True)
class DroidSettings:
    """Tunable droid behaviour; read fresh so UI-saved ``.env`` changes apply on the next read."""

    dormant_after_minutes: float
    autostart: bool
    idle_life: bool
    ntfy_server: str
    ntfy_topic: str
    voice_fx: bool
    github_repo: str
    github_token: str
    home_assistant_url: str
    home_assistant_token: str
    update_space: str
    auto_update: bool

    @classmethod
    def from_env(cls) -> "DroidSettings":
        """Read all droid settings from the environment."""
        return cls(
            dormant_after_minutes=_env_float("DROID_DORMANT_AFTER_MINUTES", 20.0),
            autostart=_env_flag("DROID_AUTOSTART", default=True),
            idle_life=_env_flag("DROID_IDLE_LIFE", default=True),
            ntfy_server=(os.getenv("DROID_NTFY_SERVER") or "https://ntfy.sh").strip().rstrip("/"),
            ntfy_topic=(os.getenv("DROID_NTFY_TOPIC") or "").strip(),
            voice_fx=_env_flag("DROID_VOICE_FX", default=True),
            github_repo=(os.getenv("DROID_GITHUB_REPO") or "").strip(),
            github_token=(os.getenv("DROID_GITHUB_TOKEN") or "").strip(),
            home_assistant_url=(os.getenv("DROID_HOME_ASSISTANT_URL") or "").strip().rstrip("/"),
            home_assistant_token=(os.getenv("DROID_HOME_ASSISTANT_TOKEN") or "").strip(),
            update_space=(os.getenv("DROID_UPDATE_SPACE") or "").strip(),
            auto_update=_env_flag("DROID_AUTO_UPDATE", default=True),
        )
