import logging
from typing import TYPE_CHECKING, Any

from reachy_mini_conversation_app.config import get_available_voices
from reachy_mini_conversation_app.droid.activation import STAGE_IDS, ActivationError, summarize, apply_answer
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


class ActivationStep(Tool):
    """Drive the droid's first-boot activation protocol one stage at a time."""

    name = "activation_step"
    description = (
        "Run your activation protocol. Actions: 'status' (current stage and what to do), 'save' (store the "
        "answer for the current stage and move on), 'redo' (go back to a stage), 'check_wake_test' (listen for "
        "the wake phrase being said), 'try_voice' (switch to a voice to demo it), 'commit' (finish activation)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "save", "redo", "check_wake_test", "try_voice", "commit"],
            },
            "value": {
                "type": "object",
                "description": "The answer for 'save' or 'try_voice', shaped as the stage instructions describe.",
            },
            "stage": {"type": "string", "enum": list(STAGE_IDS), "description": "Stage to return to for 'redo'."},
        },
        "required": ["action"],
    }

    @staticmethod
    def _stage_payload(droid: "DroidRuntime") -> dict[str, Any]:
        stage = droid.activation.stage
        identity = droid.identity
        instructions = stage.instructions.format(
            owner=identity.owner_name,
            droid_name=identity.droid_name,
            voices=", ".join(get_available_voices()),
        )
        payload: dict[str, Any] = {"stage": stage.id, "instructions": instructions}
        if stage.id == "commit":
            payload["summary"] = summarize(identity)
        return payload

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Run one activation action."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "activation is only available in the droid profile"}
        action = kwargs.get("action")
        raw_value = kwargs.get("value")
        value: dict[str, Any] = raw_value if isinstance(raw_value, dict) else {}
        stage = droid.activation.stage
        logger.info("Tool call: activation_step action=%s stage=%s", action, stage.id)

        if droid.identity.activated and action != "status":
            return {"error": "activation is already complete"}
        try:
            if action == "status":
                return self._stage_payload(droid)
            if action == "redo":
                droid.activation.go_to(str(kwargs.get("stage") or stage.id))
                return self._stage_payload(droid)
            if action == "try_voice":
                return {"status": await droid.change_voice(str(value.get("voice") or ""))}
            if action == "check_wake_test":
                if stage.id != "wake_phrase" or not droid.identity.wake_phrase:
                    return {"error": "save a wake phrase first"}
                heard = await droid.run_wake_test(droid.identity.wake_phrase)
                if heard < 2:
                    return {"passed": False, "heard": heard, "hint": "Suggest a clearer phrase and save it again."}
                droid.activation.advance()
                return {"passed": True, "heard": heard, **self._stage_payload(droid)}
            if action == "commit":
                if stage.id != "commit":
                    return {"error": f"finish stage '{stage.id}' first"}
                await droid.complete_activation()
                return {"status": "activated", "summary": summarize(droid.identity)}
            if action != "save":
                return {"error": f"unknown action {action!r}"}

            if stage.id == "boot":
                await droid.run_boot_selftest()
            elif stage.id == "owner" and not droid.face_registry.owner_enrolled():
                return {"error": "enroll the owner's face first with enroll_face role 'owner'"}
            elif stage.id == "wake_phrase":
                apply_answer(stage.id, value, droid.identity)
                unknown = droid.wake_spotter.unknown_words(droid.identity.wake_phrase)
                if unknown:
                    return {"error": f"I cannot recognise these words: {', '.join(unknown)}. Choose other words."}
                droid.save_identity()
                return {
                    "status": "phrase saved",
                    "next": "Ask your owner to say the phrase three times, then call activation_step 'check_wake_test'.",
                }
            elif stage.id == "voice":
                apply_answer(stage.id, value, droid.identity)
                await droid.change_voice(droid.identity.voice)
            else:
                apply_answer(stage.id, value, droid.identity)
        except ActivationError as e:
            return {"error": str(e)}
        droid.save_identity()
        droid.activation.advance()
        return self._stage_payload(droid)
