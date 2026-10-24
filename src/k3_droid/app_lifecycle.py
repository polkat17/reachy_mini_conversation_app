"""Helpers for app startup and shutdown lifecycle behavior."""

import asyncio
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini.utils.interpolation import distance_between_poses
from k3_droid.config import config, set_custom_profile
from k3_droid.daemon_api import DaemonApiError, daemon_request
from k3_droid.profile_store import DEFAULT_PROFILE_NAME, migrate_legacy_profiles
from k3_droid.tools.core_tools import ToolDependencies, initialize_tools
from k3_droid.tools.go_to_sleep import GoToSleep


_STOP_CURRENT_APP_PATH = "/api/apps/stop-current-app"
_STOP_CURRENT_APP_TIMEOUT_S = 2.0
_SLEEP_HEAD_TRANSLATION_TOLERANCE_M = 0.05
_SLEEP_HEAD_ROTATION_TOLERANCE_RAD = 0.35


def initialize_tools_with_default_fallback(
    instance_path: str | Path | None,
    logger: logging.Logger,
) -> str | None:
    """Load the tool registry, degrading to the default profile if need be.

    Returns the profile that was abandoned, or None when the selection loaded.
    """
    # User profiles predating profile.md were never migrated on disk; convert
    # them here so the strict readers below (and the settings UI) see them.
    try:
        migrate_legacy_profiles(config.user_personalities_root())
    except Exception as exc:
        logger.warning("Legacy profile migration failed: %s", exc)

    try:
        initialize_tools(instance_path=instance_path)
        return None
    except Exception as exc:
        selected_profile = config.REACHY_MINI_CUSTOM_PROFILE
        if not selected_profile or selected_profile == DEFAULT_PROFILE_NAME:
            raise

        # set_custom_profile is a no-op while LOCKED_PROFILE is pinned, so
        # confirm the switch took effect before announcing a fallback.
        set_custom_profile(DEFAULT_PROFILE_NAME)
        if config.REACHY_MINI_CUSTOM_PROFILE != DEFAULT_PROFILE_NAME:
            logger.error(
                "Profile %r could not be loaded (%s) and this build is locked to it, "
                "so there is no profile to fall back to.",
                selected_profile,
                exc,
            )
            raise

        logger.error(
            "Profile %r could not be loaded (%s); starting on the %r profile instead. "
            "Reselect or repair it from the settings UI to restore it.",
            selected_profile,
            exc,
            DEFAULT_PROFILE_NAME,
        )
        initialize_tools(instance_path=instance_path, force=True)
        return selected_profile


def request_stop_current_app(robot: ReachyMini, logger: logging.Logger) -> bool:
    """Request the Reachy Mini daemon to stop the current app."""
    try:
        daemon_request(
            robot,
            _STOP_CURRENT_APP_PATH,
            method="POST",
            timeout_s=_STOP_CURRENT_APP_TIMEOUT_S,
        )
    except DaemonApiError as e:
        logger.error("Failed to request current app stop: %s", e)
        return False

    logger.info("Requested current app stop via %s", _STOP_CURRENT_APP_PATH)
    return True


def _is_sleep_head_pose(head_pose: npt.ArrayLike) -> bool:
    try:
        current_head_pose: npt.NDArray[np.float64] = np.asarray(head_pose, dtype=np.float64)
    except (TypeError, ValueError):
        return False

    if current_head_pose.shape != (4, 4):
        return False

    pose_distances = distance_between_poses(current_head_pose, SLEEP_HEAD_POSE)
    translation_distance = float(pose_distances[0])
    rotation_angle = float(pose_distances[1])
    return (
        translation_distance <= _SLEEP_HEAD_TRANSLATION_TOLERANCE_M
        and rotation_angle <= _SLEEP_HEAD_ROTATION_TOLERANCE_RAD
    )


def robot_motor_state(robot: ReachyMini) -> tuple[str | None, bool | None]:
    """Return the daemon's motor mode ("enabled", "disabled", "gravity_compensation") and whether a move runs."""
    mode: object = None
    try:
        backend = getattr(robot.client.get_status(wait=True, timeout=2.0), "backend_status", None)
        mode = getattr(backend, "motor_control_mode", None)
    except Exception:
        mode = None
    if mode is None:
        try:
            mode = daemon_request(robot, "/api/motors/status").get("mode")
        except Exception:  # best effort: this only feeds a wake decision and a status line
            mode = None
    try:
        busy: bool | None = bool(daemon_request(robot, "/api/move/running"))
    except Exception:
        busy = None
    mode_text = getattr(mode, "value", mode)
    return (str(mode_text) if mode_text is not None else None), busy


def power_up_quietly(robot: ReachyMini, logger: logging.Logger) -> bool:
    """Turn torque on and glide to the neutral pose without the SDK wake-up sound.

    ``ReachyMini.wake_up()`` plays its sound through the app's own audio output, which can leave the
    conversation silent when it runs while a session is open.
    """
    try:
        robot.enable_motors()
        robot.goto_target(np.eye(4), antennas=[-0.1745, 0.1745], duration=1.5, body_yaw=0.0)
    except Exception as e:
        logger.error("Failed to power the motors up: %s", e)
        return False
    return True


def wake_up_if_sleeping(robot: ReachyMini, logger: logging.Logger) -> bool:
    """Run the SDK wake-up movement when Reachy starts from the sleep pose or with its motors off."""
    try:
        head_pose = robot.get_current_head_pose()
    except Exception as e:
        logger.warning("Could not read robot pose before startup wake-up check: %s", e)
        return False

    # The daemon leaves the motors limp after sleep and after an app stops; a limp robot ignores every target,
    # so it must be woken even when the head is not resting exactly in the sleep pose.
    motor_mode, _ = robot_motor_state(robot)
    motors_off = motor_mode is not None and motor_mode != "enabled"
    if not _is_sleep_head_pose(head_pose):
        if motors_off:
            logger.info("Motors are %s outside the sleep pose; switching them on.", motor_mode)
            return power_up_quietly(robot, logger)
        return False

    logger.info("Robot is in sleep pose; running wake-up movement.")
    try:
        robot.enable_motors()
        robot.wake_up()
    except Exception as e:
        logger.error("Failed to run wake-up movement: %s", e)
        return False
    return True


def run_go_to_sleep_tool(deps: ToolDependencies, logger: logging.Logger) -> dict[str, object]:
    """Run the shared go_to_sleep tool from synchronous shutdown paths."""
    try:
        return asyncio.run(GoToSleep()(deps))
    except Exception as e:
        logger.error("Failed to run go_to_sleep tool during shutdown: %s", e)
        return {"error": f"go_to_sleep failed: {type(e).__name__}: {e}"}
