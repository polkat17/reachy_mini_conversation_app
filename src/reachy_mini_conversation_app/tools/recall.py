import asyncio
import logging
from typing import TYPE_CHECKING, Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class Recall(Tool):
    """Search long-term memory by meaning."""

    name = "recall"
    description = (
        "Search your long-term memory (facts and past conversations) by meaning. Use it whenever a past event, "
        "person or preference might be relevant, before saying you do not know."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, e.g. 'Pasha's football team'."},
            "k": {"type": "integer", "description": "How many results, default 5."},
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return matching memories."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "memory is only available in the droid profile"}
        allowed, reason = droid.memory_access_allowed()
        if not allowed:
            return {"error": reason}
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return {"error": "query must be a non-empty string"}
        k = max(1, min(int(kwargs.get("k") or 5), 10))
        logger.info("Tool call: recall query=%s", query[:120])
        hits = await asyncio.to_thread(droid.memory.recall, query, k)
        return {"memories": [hit.text for hit in hits]} if hits else {"memories": [], "note": "nothing relevant"}
