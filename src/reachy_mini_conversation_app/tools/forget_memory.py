import asyncio
import logging
from typing import TYPE_CHECKING, Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class ForgetMemory(Tool):
    """Find and delete memories, with confirmation."""

    name = "forget_memory"
    description = (
        "Delete memories in two steps. First call with 'query' to find matches and read them back to your owner. "
        "Only after they confirm, call again with 'confirm_ids' set to the ids they approved."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to forget."},
            "confirm_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ids from the first call, like 'fact:12' or 'episode:3', that the owner confirmed.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Find candidates, or delete confirmed ids."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "memory is only available in the droid profile"}
        allowed, reason = droid.owner_access_allowed()
        if not allowed:
            return {"error": reason}
        confirm_ids = kwargs.get("confirm_ids") or []
        if confirm_ids:
            facts: list[int] = []
            episodes: list[int] = []
            for ref in confirm_ids:
                kind, _, raw_id = str(ref).partition(":")
                if raw_id.isdigit() and kind == "fact":
                    facts.append(int(raw_id))
                elif raw_id.isdigit() and kind == "episode":
                    episodes.append(int(raw_id))
            removed = droid.memory.delete_facts(facts) + droid.memory.delete_episodes(episodes)
            logger.info("Tool call: forget_memory removed=%d", removed)
            return {"status": f"deleted {removed} memories"}
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return {"error": "give a query to find memories, or confirm_ids to delete"}
        hits = await asyncio.to_thread(droid.memory.recall, query, 5, 0.35)
        if not hits:
            return {"matches": [], "note": "nothing matched; nothing was deleted"}
        return {
            "matches": [{"id": f"{hit.ref_type}:{hit.ref_id}", "text": hit.text} for hit in hits],
            "next": "Read these back and ask which to delete, then call again with confirm_ids.",
        }
