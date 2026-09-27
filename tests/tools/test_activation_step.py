from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.droid.runtime import DroidRuntime
from reachy_mini_conversation_app.droid.identity import load_identity
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.enroll_face import EnrollFace
from reachy_mini_conversation_app.tools.activation_step import ActivationStep
from reachy_mini_conversation_app.tools.background_tool_manager import ToolCallRoutine


@pytest.fixture
def deps(tmp_path: Path) -> ToolDependencies:
    """Tool dependencies wired to a droid runtime whose robot-facing steps are mocked."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), instance_path=tmp_path)
    droid = DroidRuntime(deps, tmp_path)
    droid.run_boot_selftest = AsyncMock()  # type: ignore[method-assign]
    droid.change_voice = AsyncMock(return_value="Voice changed to Ryan.")  # type: ignore[method-assign]
    droid.run_wake_test = AsyncMock(return_value=3)  # type: ignore[method-assign]
    droid.handler = MagicMock(refresh_instructions=AsyncMock())
    deps.droid = droid
    return deps


@pytest.mark.asyncio
async def test_full_activation_walkthrough(deps: ToolDependencies, tmp_path: Path) -> None:
    """Every stage in order, from boot self-test to commit, ends with an activated identity on disk."""
    step = ActivationStep()
    droid = deps.droid
    assert droid is not None

    assert (await step(deps, action="status"))["stage"] == "boot"
    assert (await step(deps, action="save"))["stage"] == "owner"
    assert "enroll the owner" in (await step(deps, action="save"))["error"]

    droid.face_registry.enroll("Pasha", [np.ones(128, dtype=np.float32) / np.sqrt(128)], is_owner=True)
    assert (await step(deps, action="save"))["stage"] == "designation"
    assert (await step(deps, action="save", value={"name": "Bolt"}))["stage"] == "wake_phrase"
    assert (await step(deps, action="save", value={"phrase": "hey bolt"}))["status"] == "phrase saved"
    assert (await step(deps, action="check_wake_test"))["passed"] is True
    await step(deps, action="try_voice", value={"voice": "Ryan"})
    assert (await step(deps, action="save", value={"voice": "Ryan"}))["stage"] == "calibration"
    await step(deps, action="save", value={"humour": "dry", "talkativeness": "brief", "language": "English"})
    await step(deps, action="save", value={"interests": ["football"], "avoid_topics": []})
    await step(deps, action="save", value={"unprompted_remarks": "rarely", "quiet_hours": "23:00-07:00"})
    await step(deps, action="save", value={"members": [{"name": "Miso", "kind": "cat"}]})
    await step(deps, action="save", value={"consent": True})
    commit_stage = await step(deps, action="save")

    assert commit_stage["stage"] == "commit" and "Bolt" in commit_stage["summary"]
    assert (await step(deps, action="commit"))["status"] == "activated"

    identity = load_identity(tmp_path)
    assert identity.activated and identity.droid_name == "Bolt" and identity.wake_phrase == "hey bolt"
    assert identity.household[0].name == "Miso"
    droid.handler.refresh_instructions.assert_awaited_once()
    assert "already complete" in (await step(deps, action="save"))["error"]


@pytest.mark.asyncio
async def test_failed_wake_test_stays_on_stage(deps: ToolDependencies) -> None:
    """A wake phrase heard fewer than twice keeps the droid on the wake phrase stage."""
    droid = deps.droid
    assert droid is not None
    droid.activation.go_to("wake_phrase")
    droid.run_wake_test = AsyncMock(return_value=1)  # type: ignore[method-assign]
    await ActivationStep()(deps, action="save", value={"phrase": "hey bolt"})

    result = await ActivationStep()(deps, action="check_wake_test")

    assert result["passed"] is False and droid.activation.stage.id == "wake_phrase"


@pytest.mark.asyncio
async def test_tools_refuse_outside_droid_profile(tmp_path: Path) -> None:
    """Without a droid runtime the tools explain themselves instead of crashing."""
    plain = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    assert "error" in await ActivationStep()(plain, action="status")
    assert "error" in await EnrollFace()(plain, role="owner")


@pytest.mark.asyncio
async def test_enroll_face_without_camera(deps: ToolDependencies) -> None:
    """With the camera disabled, enrolment reports why it cannot work."""
    result = await EnrollFace()(deps, role="owner")

    assert "camera is disabled" in result["error"]


def test_tool_call_routine_accepts_droid_deps(deps: ToolDependencies) -> None:
    """Regression: the background tool routine model must validate deps that carry a droid runtime."""
    routine = ToolCallRoutine(tool_name="activation_step", args_json_str="{}", deps=deps)

    assert routine.deps.droid is deps.droid
