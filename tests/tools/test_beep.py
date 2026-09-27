from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.beep import Beep
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.asyncio
async def test_beep_plays_rendered_file(tmp_path: Path) -> None:
    """The tool renders the pattern and hands the file to the robot speaker."""
    robot = MagicMock()
    deps = ToolDependencies(reachy_mini=robot, movement_manager=MagicMock(), instance_path=tmp_path)

    result = await Beep()(deps, pattern="affirmative")

    assert result == {"status": "played", "pattern": "affirmative"}
    played_path = Path(robot.media.play_sound.call_args.args[0])
    assert played_path.is_file() and played_path.suffix == ".wav"


@pytest.mark.asyncio
async def test_beep_rejects_unknown_pattern(tmp_path: Path) -> None:
    """Unknown patterns return an error without touching the speaker."""
    robot = MagicMock()
    deps = ToolDependencies(reachy_mini=robot, movement_manager=MagicMock(), instance_path=tmp_path)

    result = await Beep()(deps, pattern="kazoo")

    assert "error" in result
    robot.media.play_sound.assert_not_called()
