import logging
from typing import TYPE_CHECKING, Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class ListMemories(Tool):
    """List stored facts."""

    name = "list_memories"
    description = (
        "List the facts you remember, optionally only about one subject. Use for 'what do you know about me?'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"subject": {"type": "string", "description": "Only facts about this subject."}},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return stored facts."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "memory is only available in the droid profile"}
        allowed, reason = droid.memory_access_allowed()
        if not allowed:
            return {"error": reason}
        facts = droid.memory.facts(str(kwargs.get("subject") or ""), limit=40)
        logger.info("Tool call: list_memories count=%d", len(facts))
        return {"facts": [f"{fact.subject + ': ' if fact.subject else ''}{fact.text}" for fact in facts]}
