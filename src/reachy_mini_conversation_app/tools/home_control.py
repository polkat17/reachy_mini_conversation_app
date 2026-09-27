import logging
from typing import TYPE_CHECKING, Any

from reachy_mini_conversation_app.droid.home import ACTIONS, HomeError, HomeAssistant, match_device
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class HomeControl(Tool):
    """Control lights, switches, fans, scenes, scripts and media players through Home Assistant."""

    name = "home_control"
    description = (
        "Control the smart home: 'list' devices, or 'turn_on', 'turn_off' or 'toggle' one by its name (lights, "
        "switches, fans, scenes, scripts, media players). Locks and alarms are not available by design. Only act on "
        "requests from your owner."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", *ACTIONS]},
            "device": {"type": "string", "description": "The device name as the owner says it, e.g. 'desk lamp'."},
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """List devices or run an action."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "smart home control is only available in the droid profile"}
        settings = droid.settings
        if not (settings.home_assistant_url and settings.home_assistant_token):
            return {"error": "Home Assistant is not set up (DROID_HOME_ASSISTANT_URL and DROID_HOME_ASSISTANT_TOKEN)"}
        allowed, reason = droid.owner_access_allowed()
        if not allowed:
            return {"error": reason}
        action = str(kwargs.get("action") or "")
        home = HomeAssistant(settings)
        logger.info("Tool call: home_control action=%s device=%s", action, kwargs.get("device"))
        try:
            devices = await home.devices()
            if action == "list":
                return {"devices": [f"{device.name} ({device.domain}, {device.state})" for device in devices]}
            device = match_device(devices, str(kwargs.get("device") or ""))
            if device is None:
                return {"error": "no matching device", "devices": [device.name for device in devices][:30]}
            await home.call(device, action)
        except HomeError as e:
            return {"error": str(e)}
        return {"status": "done", "device": device.name, "action": action}
