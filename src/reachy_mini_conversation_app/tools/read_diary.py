import logging
from typing import TYPE_CHECKING, Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class ReadDiary(Tool):
    """Read the droid's diary."""

    name = "read_diary"
    description = "Read your private diary entry for a day (YYYY-MM-DD), or the latest one when no day is given."
    parameters_schema = {
        "type": "object",
        "properties": {"day": {"type": "string", "description": "YYYY-MM-DD, e.g. yesterday's date."}},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return the diary entry."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "the diary is only available in the droid profile"}
        allowed, reason = droid.owner_access_allowed()
        if not allowed:
            return {"error": reason}
        days = droid.memory.diary_days()
        day = str(kwargs.get("day") or (days[0] if days else "")).strip()
        entry = droid.memory.diary(day) if day else None
        logger.info("Tool call: read_diary day=%s", day)
        if entry is None:
            return {"error": "no diary entry for that day", "days_with_entries": days}
        return {"day": day, "entry": entry}
