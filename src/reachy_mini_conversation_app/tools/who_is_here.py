import logging
from typing import TYPE_CHECKING, Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class WhoIsHere(Tool):
    """Report who the droid currently recognises in view."""

    name = "who_is_here"
    description = (
        "Check who you currently see: your owner, known guests by name, or 'stranger' for unknown faces. "
        "Use it before sharing anything personal about your owner."
    )
    parameters_schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return the people in view."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "presence sensing is only available in the droid profile"}
        logger.info("Tool call: who_is_here")
        return {"in_view": droid.people_in_view(), "owner_in_view": droid.owner_in_view()}
