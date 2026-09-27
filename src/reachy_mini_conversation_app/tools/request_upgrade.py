import logging
from typing import TYPE_CHECKING, Any

import httpx

from reachy_mini_conversation_app.droid.notify import send_notification
from reachy_mini_conversation_app.droid.upgrades import (
    UpgradeRequestError,
    file_request,
    upgrades_enabled,
    list_open_requests,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class RequestUpgrade(Tool):
    """File an upgrade request so a coding agent can add a new capability."""

    name = "request_upgrade"
    description = (
        "When your owner wants something you cannot do yet, offer to file an upgrade request, and call this once "
        "they agree. Action 'list' shows open requests (check it first to avoid duplicates); action 'file' needs "
        "a short title, a clear spec of the new behaviour, and why it is wanted. Your owner approves every "
        "upgrade before it is installed."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "file"]},
            "title": {"type": "string", "description": "Five to ten words, e.g. 'Tell me when my train leaves'."},
            "spec": {"type": "string", "description": "What the droid should do, inputs and outputs, examples."},
            "reason": {"type": "string", "description": "Why the owner wants it."},
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """List or file upgrade requests."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "upgrades are only available in the droid profile"}
        settings = droid.settings
        if not upgrades_enabled(settings):
            return {"error": "the upgrade channel is not set up (DROID_GITHUB_REPO and DROID_GITHUB_TOKEN)"}
        action = kwargs.get("action")
        logger.info("Tool call: request_upgrade action=%s", action)
        try:
            if action == "list":
                requests = await list_open_requests(settings)
                return {"open_requests": [f"#{r.number} {r.title}" for r in requests]}
            title, spec = str(kwargs.get("title") or "").strip(), str(kwargs.get("spec") or "").strip()
            if not title or not spec:
                return {"error": "'file' needs a title and a spec"}
            request = await file_request(settings, droid.identity, title, spec, str(kwargs.get("reason") or ""))
        except (UpgradeRequestError, httpx.HTTPError) as e:
            logger.warning("Upgrade request failed: %s", e)
            return {"error": str(e)}
        await send_notification(
            settings, f"Upgrade request filed: {request.title}\n{request.url}", title="Droid upgrade"
        )
        return {"status": "filed", "number": request.number, "url": request.url}
