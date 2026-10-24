from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np

from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from k3_droid import daemon_api, app_lifecycle
from k3_droid.tools.core_tools import ToolDependencies


def test_request_stop_current_app_posts_to_daemon(monkeypatch) -> None:
    """The app stop request should call the connected Reachy daemon endpoint."""

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://192.168.1.42:8000/api/apps/stop-current-app"
        assert request.get_method() == "POST"
        assert timeout == 2.0
        return FakeResponse()

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)
    robot = SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))

    assert app_lifecycle.request_stop_current_app(robot, MagicMock())


def _robot(mode: str = "enabled") -> MagicMock:
    robot = MagicMock()
    robot.client.get_status.return_value = SimpleNamespace(backend_status=SimpleNamespace(motor_control_mode=mode))
    return robot


def test_wake_up_if_sleeping_enables_motors_before_wake_up() -> None:
    """Startup should enable sleeping motors before playing the wake-up movement."""
    robot = _robot()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()

    assert app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_not_called()
    assert [c for c in robot.method_calls if c in (call.enable_motors(), call.wake_up())] == [
        call.enable_motors(),
        call.wake_up(),
    ]


def test_wake_up_if_sleeping_powers_limp_motors_quietly_outside_the_sleep_pose() -> None:
    """Motors left off outside the sleep pose are switched on without the wake-up sound."""
    robot = _robot("disabled")
    robot.get_current_head_pose.return_value = np.eye(4)

    assert app_lifecycle.wake_up_if_sleeping(robot, MagicMock())
    robot.enable_motors.assert_called_once_with()
    robot.goto_target.assert_called_once()
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_skips_non_sleep_head_pose() -> None:
    """Startup should leave an already-awake robot alone."""
    robot = _robot()
    robot.get_current_head_pose.return_value = np.eye(4)

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_not_called()
    robot.enable_motors.assert_not_called()
    robot.wake_up.assert_not_called()


def test_run_go_to_sleep_tool_uses_runtime_callback() -> None:
    """Synchronous lifecycle paths should enter through the go_to_sleep tool."""
    expected = {"status": "sleeping"}
    go_to_sleep = MagicMock(return_value=expected)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )

    result = app_lifecycle.run_go_to_sleep_tool(deps, MagicMock())

    assert result == expected
    go_to_sleep.assert_called_once_with()


def test_request_stop_current_app_returns_false_on_urlerror(monkeypatch) -> None:
    """A daemon that is unreachable is reported as a failed stop, not an exception."""

    def fake_urlopen(request, timeout):
        raise daemon_api.urllib.error.URLError("connection refused")

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)
    robot = SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))

    assert not app_lifecycle.request_stop_current_app(robot, MagicMock())


def test_wake_up_if_sleeping_handles_pose_read_failure() -> None:
    """A robot that cannot report its pose skips the wake-up movement."""
    robot = _robot()
    robot.get_current_head_pose.side_effect = RuntimeError("no telemetry")

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_reports_wake_up_failure() -> None:
    """A wake-up movement that fails is reported without raising."""
    robot = _robot()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()
    robot.wake_up.side_effect = RuntimeError("motor fault")

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())
    robot.wake_up.assert_called_once_with()


def test_wake_up_if_sleeping_ignores_malformed_pose() -> None:
    """A pose that cannot be coerced to a float array is treated as not-sleeping."""
    robot = _robot()
    robot.get_current_head_pose.return_value = "not-a-pose"

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_ignores_wrong_shape_pose() -> None:
    """A pose that is not 4x4 is treated as not-sleeping."""
    robot = _robot()
    robot.get_current_head_pose.return_value = np.eye(3)

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())
    robot.wake_up.assert_not_called()


def test_run_go_to_sleep_tool_reports_tool_failure(monkeypatch) -> None:
    """A crash inside the go_to_sleep tool is surfaced as an error dict, not raised."""

    class _BoomTool:
        async def __call__(self, deps: object) -> dict[str, object]:
            raise RuntimeError("boom")

    monkeypatch.setattr(app_lifecycle, "GoToSleep", _BoomTool)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    result = app_lifecycle.run_go_to_sleep_tool(deps, MagicMock())

    assert "error" in result
    assert "boom" in str(result["error"])
