import logging
from typing import Any

from reachy_mini_conversation_app.droid.notify import send_notification, notifications_enabled
from reachy_mini_conversation_app.droid.settings import DroidSettings
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class NotifyOwner(Tool):
    """Send a push notification to the owner's phone."""

    name = "notify_owner"
    description = (
        "Send a short text message to your owner's phone. Use it when they ask you to send them something "
        "(a link, a note, a reminder) or when something important happens while they are away."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The message text, one or two sentences."},
            "urgent": {"type": "boolean", "description": "True only for things that need attention right now."},
        },
        "required": ["message"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Push the message through ntfy."""
        message = str(kwargs.get("message") or "").strip()
        if not message:
            return {"error": "message must be a non-empty string"}
        settings = DroidSettings.from_env()
        if not notifications_enabled(settings):
            return {"error": "phone notifications are not set up (DROID_NTFY_TOPIC is empty)"}
        logger.info("Tool call: notify_owner message=%s", message[:120])
        sent = await send_notification(
            settings, message, title="Droid message", priority="high" if kwargs.get("urgent") else "default"
        )
        return {"status": "sent"} if sent else {"error": "the notification service did not accept the message"}
