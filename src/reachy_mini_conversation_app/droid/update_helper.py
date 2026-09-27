"""Detached update helper: stop the droid app, install a Space revision, restart, verify, and roll back on failure.

Run by the droid itself (see ``updater.launch_update_helper``); logs go to ``<data-dir>/update.log``.
"""

import sys
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from collections.abc import Callable

import httpx
from huggingface_hub import snapshot_download

from reachy_mini_conversation_app.droid.updater import (
    HEALTH_FILENAME,
    BOOT_DORMANT_FILENAME,
    change_titles,
    load_update_state,
    save_update_state,
    health_beacon_time,
)


logger = logging.getLogger("droid.update_helper")

_STOP_TIMEOUT_S = 90.0
_HEALTH_TIMEOUT_S = 300.0
_POLL_S = 3.0


class Daemon:
    """The few daemon endpoints the helper needs."""

    def __init__(self, base_url: str) -> None:
        """Talk to the daemon at ``base_url`` (``http://host:port``)."""
        self._client = httpx.Client(base_url=base_url, timeout=30.0)

    def app_running(self, app: str) -> bool:
        """Return whether ``app`` is the running app."""
        status = self._client.get("/api/apps/current-app-status").json()
        return isinstance(status, dict) and status.get("info", {}).get("name") == app

    def stop(self) -> None:
        """Stop the current app."""
        self._client.post("/api/apps/stop-current-app").raise_for_status()

    def start(self, app: str) -> None:
        """Start ``app``."""
        self._client.post(f"/api/apps/start-app/{app}").raise_for_status()


def install_revision(space: str, revision: str, token: str | None) -> None:
    """Install a Space revision into this Python environment (the app's own venv)."""
    source = snapshot_download(space, repo_type="space", revision=revision, token=token)
    uv = shutil.which("uv")
    pip = [uv, "pip", "install", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install"]
    # Same version number on every commit, so force the package itself; then add any new dependencies.
    subprocess.run([*pip, "--force-reinstall", "--no-deps", source], check=True)
    subprocess.run([*pip, source], check=True)


def wait_until(condition: Callable[[], bool], timeout_s: float) -> bool:
    """Poll ``condition`` until it holds or ``timeout_s`` passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(_POLL_S)
    return condition()


def notify(server: str, topic: str, message: str) -> None:
    """Best-effort phone notification."""
    if not topic:
        return
    try:
        httpx.post(f"{server}/{topic}", content=message.encode("utf-8"), headers={"Title": "Droid update"}, timeout=10)
    except httpx.HTTPError as e:
        logger.warning("Notification failed: %s", e)


def run_update(args: argparse.Namespace, daemon: Daemon) -> bool:
    """Install ``args.revision``; returns True on success, False after rolling back."""
    data_dir = Path(args.data_dir)
    state = load_update_state(data_dir)
    token = args.hf_token or None

    def restart_and_check() -> bool:
        (data_dir / BOOT_DORMANT_FILENAME).touch()
        # Remove the old beacon so any beacon that appears comes from the version just started.
        (data_dir / HEALTH_FILENAME).unlink(missing_ok=True)
        daemon.start(args.app)
        return wait_until(lambda: health_beacon_time(data_dir) > 0, _HEALTH_TIMEOUT_S)

    daemon.stop()
    if not wait_until(lambda: not daemon.app_running(args.app), _STOP_TIMEOUT_S):
        logger.error("The app did not stop; aborting the update")
        return False

    try:
        install_revision(args.space, args.revision, token)
        healthy = restart_and_check()
    except (subprocess.CalledProcessError, httpx.HTTPError, OSError) as e:
        logger.error("Update to %s failed: %s", args.revision, e)
        healthy = False

    if healthy:
        titles = change_titles(args.space, token, args.previous) if args.previous else []
        state.previous_revision, state.installed_revision = args.previous, args.revision
        state.last_result, state.announce = f"installed {args.revision[:8]}", titles
        save_update_state(data_dir, state)
        notify(args.ntfy_server, args.ntfy_topic, "Upgrade installed: " + ("; ".join(titles) or args.revision[:8]))
        return True

    logger.error("New version did not report healthy; rolling back to %s", args.previous or "(unknown)")
    state.last_result = f"rolled back from {args.revision[:8]}"
    save_update_state(data_dir, state)
    if args.previous:
        try:
            if daemon.app_running(args.app):
                daemon.stop()
                wait_until(lambda: not daemon.app_running(args.app), _STOP_TIMEOUT_S)
            install_revision(args.space, args.previous, token)
            restart_and_check()
        except (subprocess.CalledProcessError, httpx.HTTPError, OSError) as e:
            logger.error("Rollback failed: %s", e)
    notify(
        args.ntfy_server, args.ntfy_topic, f"Upgrade {args.revision[:8]} failed its health check and was rolled back."
    )
    return False


def main() -> None:
    """Parse arguments and run one update."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("space", "revision", "previous", "app", "daemon-url", "data-dir"):
        parser.add_argument(f"--{name}", required=name not in ("previous",), default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--ntfy-server", default="https://ntfy.sh")
    parser.add_argument("--ntfy-topic", default="")
    args = parser.parse_args()
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(Path(args.data_dir) / "update.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    sys.exit(0 if run_update(args, Daemon(args.daemon_url)) else 1)


if __name__ == "__main__":
    main()
