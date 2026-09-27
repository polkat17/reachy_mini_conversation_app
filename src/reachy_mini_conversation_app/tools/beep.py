import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.droid.beeps import BEEP_PATTERNS, beep_file
from reachy_mini_conversation_app.droid.identity import droid_data_dir
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class Beep(Tool):
    """Play a droid beep pattern."""

    name = "beep"
    description = (
        "Play a short droid beep sequence. Use it as body language alongside speech, "
        "or on its own for a quick wordless yes, no or reaction."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "enum": sorted(BEEP_PATTERNS),
                "description": "Which beep to play: affirmative, negative, happy, sad, curious, alarm, thinking, etc.",
            },
        },
        "required": ["pattern"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Render (once) and play the requested beep."""
        pattern = kwargs.get("pattern")
        if pattern not in BEEP_PATTERNS:
            return {"error": f"unknown beep pattern {pattern!r}"}
        logger.info("Tool call: beep pattern=%s", pattern)
        try:
            path = beep_file(pattern, droid_data_dir(deps.instance_path) / "sounds")
            await asyncio.to_thread(deps.reachy_mini.media.play_sound, str(path))
        except OSError as e:
            logger.warning("Failed to play beep %s: %s", pattern, e)
            return {"error": f"failed to play beep: {e}"}
        return {"status": "played", "pattern": pattern}
