import asyncio
import logging
from typing import TYPE_CHECKING, Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class EnrollFace(Tool):
    """Learn a person's face so the droid recognises them later."""

    name = "enroll_face"
    description = (
        "Learn the face in front of the camera. Use role 'owner' during activation, or role 'guest' with the "
        "guest's name when your owner introduces someone. Tell the person to face you and turn their head "
        "slightly; it takes about five seconds. Only embeddings are stored, never pictures."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["owner", "guest"]},
            "name": {"type": "string", "description": "The guest's name. Ignored for the owner."},
        },
        "required": ["role"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Capture and store the face."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "face enrolment is only available in the droid profile"}
        is_owner = kwargs.get("role") == "owner"
        name = droid.identity.owner_name if is_owner else str(kwargs.get("name") or "").strip()
        if not name:
            return {"error": "a guest needs a name"}
        logger.info("Tool call: enroll_face name=%s owner=%s", name, is_owner)
        return await asyncio.to_thread(droid.enroll_face, name, is_owner=is_owner)
