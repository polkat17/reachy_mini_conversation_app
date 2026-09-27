import logging
from typing import TYPE_CHECKING, Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class Memorize(Tool):
    """Save a durable fact to the droid's long-term memory."""

    name = "memorize"
    description = (
        "Save one durable fact to long-term memory: when your owner says 'remember that...', or when something "
        "clearly lasting comes up (names, relationships, preferences, plans, dates). Near-duplicates update the "
        "existing fact. Never store passwords, addresses or payment details."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "One fact, third person, e.g. 'Pasha supports Arsenal'."},
            "subject": {"type": "string", "description": "Who or what it is about, e.g. 'Pasha', 'Miso the cat'."},
        },
        "required": ["text"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Store the fact."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "memory is only available in the droid profile"}
        if not droid.identity.memory_consent:
            return {"error": "memory is switched off; your owner did not consent"}
        text = str(kwargs.get("text") or "").strip()
        if not text:
            return {"error": "text must be a non-empty string"}
        logger.info("Tool call: memorize text=%s", text[:120])
        fact, created = droid.memory.remember(text, str(kwargs.get("subject") or ""))
        return {"status": "saved" if created else "updated existing fact", "fact": fact.text}
