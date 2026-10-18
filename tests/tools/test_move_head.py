from unittest.mock import MagicMock

import numpy as np
import pytest

from k3_droid.tools.move_head import MoveHead
from k3_droid.tools.core_tools import ToolDependencies
from k3_droid.dance_emotion_moves import GotoQueueMove


def _deps(body_yaw: float = 0.0, heading: float = 0.0) -> ToolDependencies:
    reachy_mini = MagicMock()
    reachy_mini.get_current_head_pose.return_value = np.eye(4)
    reachy_mini.get_current_joint_positions.return_value = ([body_yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.2])
    movement_manager = MagicMock()
    movement_manager.body_heading.return_value = heading
    movement_manager.to_body_frame.side_effect = lambda pose: pose
    return ToolDependencies(reachy_mini=reachy_mini, movement_manager=movement_manager)


@pytest.mark.asyncio
async def test_move_head_rejects_non_string_direction() -> None:
    """A non-string direction is rejected without touching the robot."""
    result = await MoveHead()(_deps(), direction=42)
    assert result == {"error": "direction must be a string"}


@pytest.mark.asyncio
async def test_move_head_queues_goto_move() -> None:
    """A valid direction queues a GotoQueueMove and marks the robot moving."""
    deps = _deps()
    result = await MoveHead()(deps, direction="left")
    assert result == {"status": "looking left"}
    queued_move = deps.movement_manager.queue_move.call_args.args[0]
    assert isinstance(queued_move, GotoQueueMove)
    deps.movement_manager.set_moving_state.assert_called_once_with(deps.motion_duration_s)


@pytest.mark.asyncio
async def test_move_head_starts_from_the_body_yaw_and_keeps_the_heading() -> None:
    """The move starts from the measured body yaw (not an antenna) relative to the heading, and ends facing it."""
    deps = _deps(body_yaw=0.9, heading=0.6)
    await MoveHead()(deps, direction="right")
    queued_move = deps.movement_manager.queue_move.call_args.args[0]
    assert queued_move.start_body_yaw == pytest.approx(0.3)
    assert queued_move.target_body_yaw == 0
    deps.movement_manager.to_body_frame.assert_called_once()


@pytest.mark.asyncio
async def test_move_head_reports_robot_failure() -> None:
    """A robot read failure is returned as an error, not raised into the loop."""
    deps = _deps()
    deps.reachy_mini.get_current_head_pose.side_effect = RuntimeError("boom")
    result = await MoveHead()(deps, direction="up")
    assert "error" in result
    assert "RuntimeError" in result["error"]
