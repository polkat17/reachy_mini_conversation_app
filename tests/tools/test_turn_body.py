from unittest.mock import MagicMock

import numpy as np
import pytest

from k3_droid.moves import MAX_BODY_HEADING_RAD
from k3_droid.tools.turn_body import TurnBody
from k3_droid.tools.core_tools import ToolDependencies


def _deps(current_target: float = 0.0) -> ToolDependencies:
    movement_manager = MagicMock()
    movement_manager.body_heading_target.return_value = current_target
    movement_manager.turn_body.side_effect = lambda heading, duration: float(
        np.clip(heading, -MAX_BODY_HEADING_RAD, MAX_BODY_HEADING_RAD)
    )
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)


@pytest.mark.asyncio
async def test_turn_body_turns_relative_to_the_current_heading() -> None:
    """Positive angles turn left from wherever the base is heading."""
    deps = _deps(current_target=np.deg2rad(30))
    result = await TurnBody()(deps, angle_degrees=60)
    heading, duration = deps.movement_manager.turn_body.call_args.args
    assert heading == pytest.approx(np.deg2rad(90))
    assert duration == pytest.approx(1.0)
    assert result == {"status": "turning to 90 degrees from the starting direction"}
    deps.movement_manager.set_moving_state.assert_called_once_with(duration)
    deps.movement_manager.set_head_tracking.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_turn_body_reports_the_turning_limit() -> None:
    """A turn past the base's limit stops at it and says so."""
    deps = _deps(current_target=np.deg2rad(-100))
    result = await TurnBody()(deps, angle_degrees=-120)
    assert result["status"] == "turning to -160 degrees from the starting direction"
    assert "limit" in result["note"]


@pytest.mark.asyncio
async def test_turn_body_face_home_returns_to_zero() -> None:
    """face_home turns back to the starting direction and ignores the angle."""
    deps = _deps(current_target=np.deg2rad(45))
    await TurnBody()(deps, face_home=True, angle_degrees=999)
    heading, _ = deps.movement_manager.turn_body.call_args.args
    assert heading == 0.0


@pytest.mark.asyncio
async def test_turn_body_rejects_a_non_numeric_angle() -> None:
    """A non-numeric angle is rejected without moving."""
    deps = _deps()
    result = await TurnBody()(deps, angle_degrees="left")
    assert result == {"error": "angle_degrees must be a number"}
    deps.movement_manager.turn_body.assert_not_called()


@pytest.mark.asyncio
async def test_turn_body_refuses_while_the_droid_is_asleep() -> None:
    """With the control loop stopped, nothing is queued to fire on wake."""
    deps = _deps()
    deps.movement_manager.is_running.return_value = False
    result = await TurnBody()(deps, angle_degrees=45)
    assert "asleep" in result["error"]
    deps.movement_manager.turn_body.assert_not_called()
