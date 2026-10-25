"""Droid settings read from the environment (documented in ``.env.example``)."""

import os
from dataclasses import dataclass

from k3_droid.config import _env_flag


# Where Pasha's weekday-morning Claude routine saves the day review; DROID_DAY_REVIEW_DATASET points elsewhere.
DEFAULT_DAY_REVIEW_DATASET = "Polkat/k3-droid-briefing"
# The Hugging Face Space this fork is published to; "Check for update" installs newer revisions from it.
DEFAULT_UPDATE_SPACE = "Polkat/k3-droid"


@dataclass(frozen=True)
class DroidSettings:
    """Tunable droid behaviour; read fresh so UI-saved ``.env`` changes apply on the next read."""

    autostart: bool
    day_review_dataset: str
    update_space: str

    @classmethod
    def from_env(cls) -> "DroidSettings":
        """Read all droid settings from the environment."""
        return cls(
            autostart=_env_flag("DROID_AUTOSTART", default=True),
            day_review_dataset=(os.getenv("DROID_DAY_REVIEW_DATASET") or DEFAULT_DAY_REVIEW_DATASET).strip(),
            update_space=(os.getenv("DROID_UPDATE_SPACE") or DEFAULT_UPDATE_SPACE).strip(),
        )
