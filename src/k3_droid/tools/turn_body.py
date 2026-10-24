import logging
from typing import Any, Dict

import numpy as np

from k3_droid.moves import MAX_BODY_HEADING_RAD
from k3_droid.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

TURN_SPEED_DEG_PER_S = 60.0
MIN_TURN_DURATION_S = 0.8


class TurnBody(Tool):
    """Rotate the whole robot on its base."""

    name = "turn_body"
    description = (
        "Turn your whole body on your rotating base, so your head and body face a new direction. "
        "angle_degrees is relative to where you face now: positive turns left, negative turns right "
        "(90 = quarter turn left, -90 = quarter turn right). Set face_home to true to face your starting direction again. "
        "Your base reaches at most 160 degrees either side of the starting direction, so you cannot turn fully around. "
        "Whenever you are asked to turn, rotate or face another direction, call this tool; never just say that you are turning. "
        "It switches face tracking off so you keep facing the new direction."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "angle_degrees": {
                "type": "number",
                "description": "Relative turn in degrees; positive is left, negative is right.",
            },
            "face_home": {
                "type": "boolean",
                "description": "Turn back to the starting direction; angle_degrees is ignored.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Queue a smooth turn of the base to the requested heading."""
        face_home = kwargs.get("face_home") is True
        angle_raw = kwargs.get("angle_degrees", 0.0)
        if not face_home and (isinstance(angle_raw, bool) or not isinstance(angle_raw, (int, float))):
            return {"error": "angle_degrees must be a number"}
        logger.info("Tool call: turn_body angle_degrees=%s face_home=%s", angle_raw, face_home)
        return turn_base(deps.movement_manager, None if face_home else float(angle_raw))


def turn_base(movement_manager: Any, angle_degrees: float | None) -> Dict[str, Any]:
    """Turn the base by ``angle_degrees`` (positive is left), or back home when None; shared with the control page."""
    # Tracking keeps the head on the person, and the robot limits how far the base may twist away from the head,
    # so it would pull the body straight back.
    movement_manager.set_head_tracking(False)
    start = movement_manager.body_heading_target()
    requested = 0.0 if angle_degrees is None else start + float(np.deg2rad(angle_degrees))
    turn_deg = abs(float(np.rad2deg(np.clip(requested, -MAX_BODY_HEADING_RAD, MAX_BODY_HEADING_RAD) - start)))
    duration = max(MIN_TURN_DURATION_S, turn_deg / TURN_SPEED_DEG_PER_S)
    heading = movement_manager.turn_body(requested, duration)
    movement_manager.set_moving_state(duration)

    heading_deg = round(float(np.rad2deg(heading)))
    result: Dict[str, Any] = {"status": f"turning to {heading_deg} degrees from the starting direction"}
    if abs(heading - requested) > 1e-6:
        result["note"] = "reached the base's turning limit; stopped short of the requested angle"
    return result
