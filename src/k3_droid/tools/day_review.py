import asyncio
import logging
from typing import TYPE_CHECKING, Any

from k3_droid.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from k3_droid.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class DayReview(Tool):
    """Read today's prioritised review of the owner's meetings and email."""

    name = "day_review"
    description = (
        "Read today's review of your owner's day, prepared each weekday morning from their calendar and email: "
        "prioritised meetings with what to prepare, emails to answer, and warnings such as clashes. "
        "Use it when they ask about their plan, meetings, schedule or what to focus on today."
    )
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return today's review."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "the day review is only available in the droid profile"}
        allowed, reason = droid.owner_access_allowed()
        if not allowed:
            return {"error": reason}
        logger.info("Tool call: day_review")
        review = await asyncio.to_thread(droid.day_review)
        if review is None:
            return {
                "error": "no review for today yet (it is prepared on weekday mornings; set DROID_DAY_REVIEW_DATASET)"
            }
        return review
