import argparse
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.droid import updater as updater_mod
from reachy_mini_conversation_app.droid import update_helper as helper_mod
from reachy_mini_conversation_app.droid.updater import (
    UpdateState,
    load_update_state,
    save_update_state,
    health_beacon_time,
    write_health_beacon,
    take_boot_dormant_flag,
)


def test_state_beacon_and_boot_flag(tmp_path: Path) -> None:
    """Update state round-trips, the beacon records boot time, and the boot flag is consumed once."""
    save_update_state(tmp_path, UpdateState(installed_revision="abc", announce=["New: trains"]))
    write_health_beacon(tmp_path)
    (tmp_path / updater_mod.BOOT_DORMANT_FILENAME).touch()

    assert load_update_state(tmp_path).announce == ["New: trains"]
    assert health_beacon_time(tmp_path) > 0
    assert take_boot_dormant_flag(tmp_path) is True and take_boot_dormant_flag(tmp_path) is False


class _FakeDaemon:
    def __init__(self, data_dir: Path, healthy_revisions: set[str], installed: list[str]) -> None:
        self.running = True
        self.data_dir = data_dir
        self.healthy_revisions = healthy_revisions
        self.installed = installed
        self.starts = 0

    def app_running(self, _app: str) -> bool:
        return self.running

    def stop(self) -> None:
        self.running = False

    def start(self, _app: str) -> None:
        self.running = True
        self.starts += 1
        if self.installed[-1] in self.healthy_revisions:
            write_health_beacon(self.data_dir)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        space="owner/droid",
        revision="new-sha",
        previous="old-sha",
        app="droid_app",
        data_dir=str(tmp_path),
        hf_token="",
        ntfy_server="https://ntfy.sh",
        ntfy_topic="",
    )


@pytest.fixture
def fast_helper(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make the helper poll quickly and record installs instead of running pip; returns the install log."""
    installed: list[str] = []
    monkeypatch.setattr(helper_mod, "_POLL_S", 0.01)
    monkeypatch.setattr(helper_mod, "_HEALTH_TIMEOUT_S", 0.2)
    monkeypatch.setattr(helper_mod, "install_revision", lambda _space, revision, _token: installed.append(revision))
    monkeypatch.setattr(helper_mod, "change_titles", lambda *_args: ["Add train announcements"])
    return installed


def test_healthy_update_is_kept(tmp_path: Path, fast_helper: list[str]) -> None:
    """A new version that reports healthy becomes the installed revision and is announced on wake."""
    daemon = _FakeDaemon(tmp_path, {"new-sha"}, fast_helper)

    assert helper_mod.run_update(_args(tmp_path), daemon) is True  # type: ignore[arg-type]

    state = load_update_state(tmp_path)
    assert fast_helper == ["new-sha"] and state.installed_revision == "new-sha"
    assert state.previous_revision == "old-sha" and state.announce == ["Add train announcements"]
    assert (tmp_path / updater_mod.BOOT_DORMANT_FILENAME).exists()


def test_unhealthy_update_rolls_back(tmp_path: Path, fast_helper: list[str]) -> None:
    """Without a health beacon the previous revision is reinstalled and restarted."""
    daemon = _FakeDaemon(tmp_path, {"old-sha"}, fast_helper)

    assert helper_mod.run_update(_args(tmp_path), daemon) is False  # type: ignore[arg-type]

    assert fast_helper == ["new-sha", "old-sha"] and daemon.starts == 2
    assert load_update_state(tmp_path).last_result.startswith("rolled back")


def test_failed_install_rolls_back(tmp_path: Path, fast_helper: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """A pip failure on the new revision also rolls back."""

    def install(_space: str, revision: str, _token: str | None) -> None:
        fast_helper.append(revision)
        if revision == "new-sha":
            raise subprocess.CalledProcessError(1, "pip")

    monkeypatch.setattr(helper_mod, "install_revision", install)
    daemon = _FakeDaemon(tmp_path, {"old-sha"}, fast_helper)

    assert helper_mod.run_update(_args(tmp_path), daemon) is False  # type: ignore[arg-type]
    assert fast_helper == ["new-sha", "old-sha"]


def test_launch_helper_is_detached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper runs in its own session so stopping the app does not kill it."""
    popen = MagicMock()
    monkeypatch.setattr(updater_mod.subprocess, "Popen", popen)

    updater_mod.launch_update_helper({"space": "owner/droid", "data_dir": "/tmp/x"})

    command = popen.call_args.args[0]
    assert command[1:3] == ["-m", "reachy_mini_conversation_app.droid.update_helper"]
    assert "--data-dir" in command and popen.call_args.kwargs["start_new_session"] is True
