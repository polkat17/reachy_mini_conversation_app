"""The droid runtime: one object that listens to the conversation and drives droid behaviour."""

import time
import queue
import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any
from pathlib import Path
from datetime import datetime
from collections.abc import Callable, Awaitable, Coroutine

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.streaming import AudioArray
from reachy_mini_conversation_app.tools.beep import Beep
from reachy_mini_conversation_app.droid.audio import LoudnessTrigger, to_mono_16k
from reachy_mini_conversation_app.droid.emotes import EmoteGuard
from reachy_mini_conversation_app.droid.identity import Identity, load_identity
from reachy_mini_conversation_app.droid.settings import DroidSettings
from reachy_mini_conversation_app.droid.autostart import ensure_autostart
from reachy_mini_conversation_app.droid.idle_life import IdleLife, idle_remark_prompt
from reachy_mini_conversation_app.tools.move_head import MoveHead
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion
from reachy_mini_conversation_app.conversation_handler import ConversationEvent, ConversationHandler


if TYPE_CHECKING:
    from reachy_mini_conversation_app.console import LocalStream


logger = logging.getLogger(__name__)

AudioListener = Callable[[NDArray[np.float32]], None]
LifecycleHook = Callable[[str], Awaitable[None]]

_TICK_INTERVAL_S = 1.0
_GOODBYE_GRACE_S = 4.0
_AUDIO_QUEUE_FRAMES = 200


class DroidRuntime:
    """Owns the droid's behaviours and follows whichever conversation handler is active."""

    def __init__(self, deps: ToolDependencies, instance_path: str | Path | None) -> None:
        """Create the behaviours; nothing runs until ``start`` and ``attach`` are called."""
        self.deps = deps
        self.instance_path = instance_path
        self.settings = DroidSettings.from_env()
        self.identity: Identity = load_identity(instance_path)
        self.handler: ConversationHandler | None = None
        self.stream: "LocalStream | None" = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._stop = threading.Event()
        self._transition_lock = asyncio.Lock()
        self._emotion_tool = PlayEmotion()
        self._beep_tool = Beep()
        self._move_head_tool = MoveHead()
        self._emote_guard = EmoteGuard(self._queue_emotion)
        self._idle_life = IdleLife()
        self._last_user_turn_at = time.monotonic()
        self._loudness_trigger = LoudnessTrigger()
        self._audio_queue: queue.Queue[NDArray[np.float32]] = queue.Queue(maxsize=_AUDIO_QUEUE_FRAMES)
        self._audio_listeners: list[AudioListener] = [self._wake_on_loud_speech]
        self._event_listeners: list[Callable[[ConversationEvent], None]] = [self._emote_guard.on_event]
        self.on_dormant: list[LifecycleHook] = []
        self.on_wake: list[LifecycleHook] = []

    # ---- wiring -------------------------------------------------------------

    def attach(self, handler: ConversationHandler) -> None:
        """Listen to ``handler``; called for every handler the app builds."""
        self.handler = handler
        handler.set_conversation_listener(self._on_event)

    def bind_stream(self, stream: "LocalStream") -> None:
        """Take over the stream's mic tap; the stream owns the session lifecycle."""
        self.stream = stream
        stream.audio_tap = self._tap_audio

    def start(self) -> None:
        """Start background work: autostart registration, audio listeners, and the tick loop."""
        if self.settings.autostart:
            threading.Thread(
                target=ensure_autostart, args=(self.deps.reachy_mini,), daemon=True, name="droid-autostart"
            ).start()
        threading.Thread(target=self._audio_worker, daemon=True, name="droid-audio").start()
        threading.Thread(target=self._start_tick_loop_when_ready, daemon=True, name="droid-tick-start").start()

    def stop(self) -> None:
        """Stop background threads."""
        self._stop.set()

    def add_audio_listener(self, listener: AudioListener) -> None:
        """Receive every mic frame as 16 kHz mono float32, on the droid audio thread."""
        self._audio_listeners.append(listener)

    def add_event_listener(self, listener: Callable[[ConversationEvent], None]) -> None:
        """Receive every conversation milestone, on the event loop."""
        self._event_listeners.append(listener)

    def reload_identity(self) -> Identity:
        """Re-read ``core.yaml`` after it changed."""
        self.identity = load_identity(self.instance_path)
        return self.identity

    @property
    def is_dormant(self) -> bool:
        """Return whether the droid is asleep with its realtime session closed."""
        return self.stream is not None and self.stream.is_dormant

    # ---- event loop helpers --------------------------------------------------

    def _on_event(self, event: ConversationEvent) -> None:
        self._loop = asyncio.get_running_loop()
        if event.kind == "user_transcript":
            self._last_user_turn_at = time.monotonic()
        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning("Droid event listener failed on %s: %s", event.kind, e)

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        if self._loop is None:
            coro.close()
            return
        task = self._loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def submit(self, coro: Coroutine[Any, Any, Any]) -> bool:
        """Run ``coro`` on the app event loop from any thread; returns False when the loop is not running yet."""
        loop = self._loop
        if loop is None or not loop.is_running():
            coro.close()
            return False
        asyncio.run_coroutine_threadsafe(coro, loop)
        return True

    def _start_tick_loop_when_ready(self) -> None:
        while not self._stop.is_set():
            stream = self.stream
            loop = stream.asyncio_loop if stream is not None else None
            if loop is not None and loop.is_running():
                self._loop = loop
                asyncio.run_coroutine_threadsafe(self._tick_loop(), loop)
                return
            time.sleep(0.2)

    # ---- speech --------------------------------------------------------------

    async def say_event(self, text: str) -> bool:
        """Voice a ``[SYSTEM EVENT]`` prompt now; returns False when no session is open."""
        handler = self.handler
        if handler is None or not handler._is_connected():
            return False
        try:
            await handler.say(text)
        except (RuntimeError, ValueError) as e:
            logger.warning("Droid could not speak event: %s", e)
            return False
        return True

    def _queue_emotion(self, emotion: str) -> None:
        logger.info("Droid fallback emote: %s", emotion)
        self._spawn(self._emotion_tool(self.deps, emotion=emotion))

    async def beep(self, pattern: str) -> None:
        """Play a beep pattern, logging failures."""
        result = await self._beep_tool(self.deps, pattern=pattern)
        if "error" in result:
            logger.warning("Droid beep failed: %s", result["error"])

    # ---- dormant / wake -------------------------------------------------------

    def request_dormant(self, reason: str = "requested") -> dict[str, Any]:
        """Thread-safe: put the droid to sleep without stopping the app (the ``go_to_sleep`` tool path)."""
        if not self.submit(self.enter_dormant(reason)):
            return {"error": "the droid runtime is not running yet"}
        return {"status": "going_dormant", "note": "Say a short goodnight; your session will close in a moment."}

    def request_wake(self, reason: str = "requested") -> dict[str, Any]:
        """Thread-safe: wake the droid and open a new session."""
        if not self.submit(self.wake(reason)):
            return {"error": "the droid runtime is not running yet"}
        return {"status": "waking"}

    async def enter_dormant(self, reason: str) -> None:
        """Say goodnight, run dormant hooks, move to the sleep pose and close the session."""
        stream = self.stream
        if stream is None:
            return
        async with self._transition_lock:
            if stream.is_dormant:
                return
            logger.info("Droid going dormant (%s)", reason)
            await asyncio.sleep(_GOODBYE_GRACE_S)
            await self._run_hooks(self.on_dormant, reason)
            await self.beep("shutdown")
            await asyncio.to_thread(self._sleep_body)
            await stream.enter_dormant()

    async def wake(self, reason: str) -> None:
        """Wake the body, reopen the session and run wake hooks."""
        stream = self.stream
        if stream is None:
            return
        async with self._transition_lock:
            if not stream.is_dormant:
                return
            logger.info("Droid waking up (%s)", reason)
            self.reload_identity()
            await asyncio.to_thread(self._wake_body)
            await self.beep("boot")
            self._last_user_turn_at = time.monotonic()
            await stream.wake()
            await self._run_hooks(self.on_wake, reason)

    async def _run_hooks(self, hooks: list[LifecycleHook], reason: str) -> None:
        for hook in hooks:
            try:
                await hook(reason)
            except Exception as e:
                logger.warning("Droid lifecycle hook failed (%s): %s", reason, e)

    def _sleep_body(self) -> None:
        robot = self.deps.reachy_mini
        self.deps.movement_manager.stop(reset_to_neutral=False)
        try:
            robot.disable_wobbling()
            robot.goto_sleep()
        except Exception as e:
            logger.error("Failed to move to the sleep pose: %s", e)

    def _wake_body(self) -> None:
        robot = self.deps.reachy_mini
        try:
            robot.enable_motors()
            robot.wake_up()
            robot.enable_wobbling()
        except Exception as e:
            logger.error("Failed to run the wake-up movement: %s", e)
        self.deps.movement_manager.start()

    # ---- audio ----------------------------------------------------------------

    def _tap_audio(self, sample_rate: int, frame: AudioArray) -> None:
        try:
            self._audio_queue.put_nowait(to_mono_16k(sample_rate, frame))
        except queue.Full:
            logger.debug("Droid audio queue full; dropping a frame")

    def _audio_worker(self) -> None:
        while not self._stop.is_set():
            try:
                samples = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            for listener in self._audio_listeners:
                try:
                    listener(samples)
                except Exception as e:
                    logger.warning("Droid audio listener failed: %s", e)

    def _wake_on_loud_speech(self, samples: NDArray[np.float32]) -> None:
        # Until a wake phrase is chosen during activation, any sustained speech wakes the droid.
        if not self.is_dormant or self.identity.wake_phrase:
            self._loudness_trigger.reset()
            return
        if self._loudness_trigger.feed(samples):
            self.request_wake("speech")

    # ---- periodic behaviour --------------------------------------------------

    async def _tick_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception as e:
                logger.warning("Droid tick failed: %s", e)
            await asyncio.sleep(_TICK_INTERVAL_S)

    def seconds_since_activity(self) -> float:
        """Seconds since the conversation last saw any activity, including the droid's own speech."""
        handler = self.handler
        return float("inf") if handler is None else time.monotonic() - handler.last_activity_time

    def seconds_since_user_turn(self) -> float:
        """Seconds since someone last spoke to the droid (or since it last woke up)."""
        return time.monotonic() - self._last_user_turn_at

    def someone_present(self) -> bool:
        """Return whether a person is currently in view; overridden once presence sensing is running."""
        return False

    async def tick(self) -> None:
        """Run once per second: dormancy timeout and idle life."""
        if self.stream is None or self.is_dormant:
            return
        idle_s = self.seconds_since_activity()
        dormant_after_s = self.settings.dormant_after_minutes * 60.0
        if dormant_after_s > 0 and self.seconds_since_user_turn() >= dormant_after_s and not self.someone_present():
            self._spawn(self.enter_dormant("inactivity"))
            return
        if not self.settings.idle_life or not self.deps.movement_manager.is_idle():
            return
        action = self._idle_life.next_action(idle_s, time.monotonic(), self.identity.protocols, datetime.now())
        if action is None:
            return
        logger.info("Droid idle life: %s %s", action.kind, action.value)
        if action.kind == "beep":
            await self.beep(action.value)
        elif action.kind == "emotion":
            await self._emotion_tool(self.deps, emotion=action.value)
        elif action.kind == "look":
            await self._move_head_tool(self.deps, direction=action.value)
        elif action.kind == "remark":
            await self.say_event(idle_remark_prompt(idle_s / 60.0))
