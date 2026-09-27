import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.tools import home_control as home_control_mod
from reachy_mini_conversation_app.tools import request_upgrade as request_upgrade_mod
from reachy_mini_conversation_app.droid.home import Device
from reachy_mini_conversation_app.droid.settings import DroidSettings
from reachy_mini_conversation_app.droid.upgrades import UpgradeRequest
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps(tmp_path: Path, **settings: str) -> ToolDependencies:
    droid = MagicMock()
    droid.settings = dataclasses.replace(DroidSettings.from_env(), **settings)
    droid.owner_access_allowed.return_value = (True, "")
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), instance_path=tmp_path, droid=droid)


@pytest.mark.asyncio
async def test_request_upgrade_requires_setup(tmp_path: Path) -> None:
    """Without a repository and token the tool explains what is missing."""
    result = await request_upgrade_mod.RequestUpgrade()(_deps(tmp_path), action="list")

    assert "DROID_GITHUB_REPO" in result["error"]


@pytest.mark.asyncio
async def test_request_upgrade_files_and_notifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Filing returns the issue and pings the owner's phone."""
    monkeypatch.setattr(
        request_upgrade_mod, "file_request", AsyncMock(return_value=UpgradeRequest(3, "Trains", "https://x/3"))
    )
    notify = AsyncMock(return_value=True)
    monkeypatch.setattr(request_upgrade_mod, "send_notification", notify)
    deps = _deps(tmp_path, github_repo="owner/fork", github_token="t")

    result = await request_upgrade_mod.RequestUpgrade()(deps, action="file", title="Trains", spec="Announce trains")

    assert result == {"status": "filed", "number": 3, "url": "https://x/3"}
    notify.assert_awaited_once()


@pytest.mark.asyncio
async def test_home_control_locked_for_guests(tmp_path: Path) -> None:
    """With only a guest in view, the home stays under the owner's control."""
    deps = _deps(tmp_path, home_assistant_url="http://ha", home_assistant_token="t")
    deps.droid.owner_access_allowed.return_value = (False, "Pasha is not in view")

    result = await home_control_mod.HomeControl()(deps, action="turn_on", device="lamp")

    assert result == {"error": "Pasha is not in view"}


@pytest.mark.asyncio
async def test_home_control_turns_on_matched_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The spoken name is matched and the action forwarded."""
    home = MagicMock()
    home.devices = AsyncMock(return_value=[Device("light.desk_lamp", "Desk lamp", "off")])
    home.call = AsyncMock()
    monkeypatch.setattr(home_control_mod, "HomeAssistant", MagicMock(return_value=home))
    deps = _deps(tmp_path, home_assistant_url="http://ha", home_assistant_token="t")

    result = await home_control_mod.HomeControl()(deps, action="turn_on", device="desk lamp")

    assert result == {"status": "done", "device": "Desk lamp", "action": "turn_on"}
    home.call.assert_awaited_once()
