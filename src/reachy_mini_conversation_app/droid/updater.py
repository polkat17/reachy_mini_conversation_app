"""Self-development, part 2: install merged upgrades from the droid's Hugging Face Space, with rollback.

The fork's ``droid-sync-space`` workflow mirrors ``main`` to a Space. While the droid is dormant at night, it
compares the Space's latest revision with the installed one and, when they differ, starts ``update_helper`` as a
detached process. The helper stops the app through the daemon, installs the new revision, starts the app again,
and waits for the new version's health beacon; without one it reinstalls the previous revision.
"""

import os
import sys
import json
import time
import logging
import subprocess
from typing import Any
from pathlib import Path
from dataclasses import field, asdict, dataclass

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


logger = logging.getLogger(__name__)

UPDATE_STATE_FILENAME = "update.json"
HEALTH_FILENAME = "health.json"
BOOT_DORMANT_FILENAME = "boot_dormant"


@dataclass
class UpdateState:
    """What is installed, what was installed before, and how the last update went."""

    installed_revision: str = ""
    previous_revision: str = ""
    last_result: str = ""
    last_checked_at: float = 0.0
    announce: list[str] = field(default_factory=list)


def load_update_state(data_dir: Path) -> UpdateState:
    """Read ``update.json``; missing or unreadable files give an empty state."""
    try:
        raw = json.loads((data_dir / UPDATE_STATE_FILENAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return UpdateState()
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable update state: %s", e)
        return UpdateState()
    if not isinstance(raw, dict):
        return UpdateState()
    known = {key: raw[key] for key in UpdateState.__dataclass_fields__ if key in raw}
    return UpdateState(**known)


def save_update_state(data_dir: Path, state: UpdateState) -> None:
    """Write ``update.json``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / UPDATE_STATE_FILENAME).write_text(json.dumps(asdict(state)), encoding="utf-8")


def write_health_beacon(data_dir: Path) -> None:
    """Record that this version booted and runs its main loop (the helper's success signal)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / HEALTH_FILENAME).write_text(
        json.dumps({"booted_at": time.time(), "pid": os.getpid()}), encoding="utf-8"
    )


def health_beacon_time(data_dir: Path) -> float:
    """Return when the last health beacon was written, or 0."""
    try:
        return float(json.loads((data_dir / HEALTH_FILENAME).read_text(encoding="utf-8"))["booted_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def take_boot_dormant_flag(data_dir: Path) -> bool:
    """Return whether the app was restarted by an update (and should stay dormant), clearing the flag."""
    flag = data_dir / BOOT_DORMANT_FILENAME
    if not flag.exists():
        return False
    flag.unlink(missing_ok=True)
    return True


def latest_revision(space: str, token: str | None) -> str | None:
    """Return the Space's current commit sha, or None when it cannot be read."""
    try:
        return HfApi(token=token).space_info(space).sha
    except (HfHubHTTPError, OSError) as e:
        logger.warning("Could not check %s for updates: %s", space, e)
        return None


def change_titles(space: str, token: str | None, since_revision: str, limit: int = 5) -> list[str]:
    """Return commit titles newer than ``since_revision``, newest first."""
    try:
        commits = HfApi(token=token).list_repo_commits(space, repo_type="space")
    except (HfHubHTTPError, OSError) as e:
        logger.warning("Could not list changes in %s: %s", space, e)
        return []
    titles: list[str] = []
    for commit in commits:
        if commit.commit_id == since_revision:
            break
        titles.append(commit.title)
    return titles[:limit]


def launch_update_helper(args: dict[str, Any]) -> None:
    """Start ``update_helper`` detached from this process, so it survives the app being stopped."""
    command = [sys.executable, "-m", "reachy_mini_conversation_app.droid.update_helper"]
    for key, value in args.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    logger.info("Launching update helper: %s", " ".join(command))
    subprocess.Popen(command, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
