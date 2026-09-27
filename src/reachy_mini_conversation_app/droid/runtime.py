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

from reachy_mini_conversation_app.memory import list_memory_facts
from reachy_mini_conversation_app.streaming import AudioArray
from reachy_mini_conversation_app.tools.beep import Beep
from reachy_mini_conversation_app.droid.audio import LoudnessTrigger, to_mono_16k
from reachy_mini_conversation_app.droid.faces import FaceRegistry
from reachy_mini_conversation_app.tools.dance import Dance
from reachy_mini_conversation_app.droid.emotes import EmoteGuard
from reachy_mini_conversation_app.droid.notify import send_notification
from reachy_mini_conversation_app.droid.senses import SoundEvent, SoundWatcher, SoundClassifier
from reachy_mini_conversation_app.droid.context import OWNER_LAST_SEEN_KEY, mood_for, open_memory
from reachy_mini_conversation_app.droid.journal import (
    CHECKPOINT_TURNS,
    SessionJournal,
    summarize_session,
    write_diary_entry,
)
from reachy_mini_conversation_app.droid.identity import Identity, load_identity, save_identity, droid_data_dir
from reachy_mini_conversation_app.droid.presence import CAT, STRANGER, PresenceEvent, PresenceSensor
from reachy_mini_conversation_app.droid.settings import DroidSettings
from reachy_mini_conversation_app.droid.voice_fx import DroidVoiceFilter
from reachy_mini_conversation_app.droid.autostart import ensure_autostart
from reachy_mini_conversation_app.droid.idle_life import IdleLife, in_quiet_hours, idle_remark_prompt
from reachy_mini_conversation_app.droid.reactions import cat_prompt, presence_prompt, break_nudge_prompt
from reachy_mini_conversation_app.tools.move_head import MoveHead
from reachy_mini_conversation_app.droid.activation import ActivationProgress
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.stop_dance import StopDance
from reachy_mini_conversation_app.droid.wake_phrase import WakePhraseSpotter
from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove
from reachy_mini_conversation_app.conversation_handler import ConversationEvent, ConversationHandler


if TYPE_CHECKING:
    from reachy_mini_conversation_app.console import LocalStream


logger = logging.getLogger(__name__)

AudioListener = Callable[[NDArray[np.float32]], None]
LifecycleHook = Callable[[str], Awaitable[None]]

_TICK_INTERVAL_S = 1.0
_GOODBYE_GRACE_S = 4.0
_AUDIO_QUEUE_FRAMES = 200
_PRESENCE_INTERVAL_AWAKE_S = 1.0
_PRESENCE_INTERVAL_DORMANT_S = 1.5
_REMINDER_SESSION_GRACE_S = 60.0
_CAT_REACTION_COOLDOWN_S = 600.0
_TALKING_RECENTLY_S = 10.0


class DroidRuntime:
    """Owns the droid's behaviours and follows whichever conversation handler is active."""

    def __init__(self, deps: ToolDependencies, instance_path: str | Path | None) -> None:
        """Create the behaviours; nothing runs until ``start`` and ``attach`` are called."""
        self.deps = deps
        self.instance_path = instance_path
        self.settings = DroidSettings.from_env()
        self.identity: Identity = load_identity(instance_path)
        self.data_dir = droid_data_dir(instance_path)
        self.face_registry = FaceRegistry(self.data_dir)
        self.presence: PresenceSensor | None = None
        self.wake_spotter = WakePhraseSpotter(self.data_dir / "models")
        self.activation = ActivationProgress(self.data_dir)
        self._wake_test_active = False
        self._wake_test_hits = 0
        self._last_break_nudge_at = time.monotonic()
        self.memory = open_memory(instance_path)
        self._sound_watcher: SoundWatcher | None = None
        self._sound_loading = False
        self._voice_filters: dict[int, DroidVoiceFilter] = {}
        self._last_cat_reaction = float("-inf")
        self._dance_tool = Dance()
        self._stop_dance_tool = StopDance()
        self.journal = SessionJournal()
        self._checkpointing = False
        self._import_legacy_facts()
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
        self._audio_listeners: list[AudioListener] = [
            self._wake_on_loud_speech,
            self._listen_for_wake_phrase,
            self._listen_for_sounds,
        ]
        self._event_listeners: list[Callable[[ConversationEvent], None]] = [
            self._emote_guard.on_event,
            self._on_session_started,
            self._record_transcript,
        ]
        self.on_dormant: list[LifecycleHook] = [self._checkpoint_on_dormant]
        self.on_wake: list[LifecycleHook] = []
        self.on_factory_reset: list[LifecycleHook] = [self._wipe_memory]

    # ---- wiring -------------------------------------------------------------

    def attach(self, handler: ConversationHandler) -> None:
        """Listen to ``handler``; called for every handler the app builds."""
        self.handler = handler
        handler.set_conversation_listener(self._on_event)

    def bind_stream(self, stream: "LocalStream") -> None:
        """Take over the stream's mic tap; the stream owns the session lifecycle."""
        self.stream = stream
        stream.audio_tap = self._tap_audio
        if self.settings.voice_fx:
            stream.output_filter = self._apply_voice_fx

    def start(self) -> None:
        """Start background work: autostart registration, audio listeners, and the tick loop."""
        if self.settings.autostart:
            threading.Thread(
                target=ensure_autostart, args=(self.deps.reachy_mini,), daemon=True, name="droid-autostart"
            ).start()
        threading.Thread(target=self._audio_worker, daemon=True, name="droid-audio").start()
        if self.identity.wake_phrase:
            self.wake_spotter.set_phrase(self.identity.wake_phrase)
        if self.deps.camera_enabled:
            self.presence = PresenceSensor(
                self.deps.reachy_mini.media.get_frame,
                self.face_registry,
                self._on_presence_event,
                lambda: _PRESENCE_INTERVAL_DORMANT_S if self.is_dormant else _PRESENCE_INTERVAL_AWAKE_S,
            )
            self.presence.start()
        threading.Thread(target=self._start_tick_loop_when_ready, daemon=True, name="droid-tick-start").start()

    def stop(self) -> None:
        """Stop background threads."""
        self._stop.set()
        if self.presence is not None:
            self.presence.stop()

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

    def save_identity(self) -> None:
        """Persist the in-memory identity to ``core.yaml``."""
        save_identity(self.instance_path, self.identity)

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
        """Return whether any person is currently in view."""
        return bool(self.people_in_view())

    def owner_in_view(self) -> bool:
        """Return whether the owner is currently recognised in view."""
        if self.presence is None:
            return False
        present = set(self.presence.tracker.present())
        return any(person.is_owner and person.name in present for person in self.face_registry.people())

    def people_in_view(self) -> list[str]:
        """Return the people in view (``stranger`` for unrecognised faces); the cat is not a person."""
        return [] if self.presence is None else [name for name in self.presence.tracker.present() if name != CAT]

    async def tick(self) -> None:
        """Run once per second: reminders, dormancy timeout, briefings, diary and idle life."""
        if self.stream is None:
            return
        if self.is_dormant:
            if self.memory.due_reminders(time.time()):
                await self.wake("reminder")
            return
        idle_s = self.seconds_since_activity()
        dormant_after_s = self.settings.dormant_after_minutes * 60.0
        if dormant_after_s > 0 and self.seconds_since_user_turn() >= dormant_after_s and not self.someone_present():
            self._spawn(self.enter_dormant("inactivity"))
            return
        await self._maybe_nudge_break()
        await self._deliver_due_reminders()
        await self._maybe_morning_briefing()
        await self._maybe_write_diary()
        await self._maybe_keep_dancing()
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

    async def _maybe_nudge_break(self) -> None:
        minutes = self.identity.protocols.break_nudge_minutes
        if self.presence is None or not self.identity.activated or minutes <= 0:
            return
        now = time.monotonic()
        present_s = self.presence.tracker.owner_present_for(now)
        if present_s >= minutes * 60 and now - self._last_break_nudge_at >= minutes * 60:
            self._last_break_nudge_at = now
            await self.say_event(break_nudge_prompt(self.identity, present_s))

    # ---- presence --------------------------------------------------------------

    def _on_presence_event(self, event: PresenceEvent) -> None:
        self.submit(self._react_to_presence(event))

    async def _react_to_presence(self, event: PresenceEvent) -> None:
        if event.is_owner:
            self.memory.set_state(OWNER_LAST_SEEN_KEY, str(time.time()))
        if not self.identity.activated:
            return
        if self.is_dormant:
            if event.kind == "arrived" and event.is_owner and self.identity.protocols.greet_on_sight:
                await self.wake("owner_seen")
            return
        if in_quiet_hours(self.identity.protocols.quiet_hours, datetime.now()):
            return
        if event.name == CAT:
            if event.kind == "arrived":
                await self.react_to_cat(heard=False)
            return
        if event.kind == "arrived" and event.name == STRANGER:
            await self._emotion_tool(self.deps, emotion="startled")
        prompt = presence_prompt(event, self.identity)
        if prompt is not None:
            await self.say_event(prompt)

    def enroll_face(self, name: str, *, is_owner: bool) -> dict[str, Any]:
        """Blocking: capture the largest face for a few seconds and store its embeddings under ``name``."""
        if self.presence is None:
            return {"error": "the camera is disabled, so faces cannot be enrolled"}
        embeddings = self.presence.capture_embeddings(exclude_known=not is_owner)
        if len(embeddings) < 3:
            return {"error": "I could not see a face clearly; ask them to face the camera in good light and retry"}
        stored = self.face_registry.enroll(name, embeddings, is_owner=is_owner)
        return {"status": "enrolled", "name": name, "samples": stored}

    # ---- activation --------------------------------------------------------------

    def _on_session_started(self, event: ConversationEvent) -> None:
        if event.kind == "session_started" and self.identity.protocols.follow_face and self.deps.camera_enabled:
            self.deps.movement_manager.set_head_tracking(True)

    async def run_boot_selftest(self) -> None:
        """Boot self-test: beeps, a head scan and an antenna twitch."""
        await self.beep("boot")
        for direction in ("left", "right", "front"):
            await self._move_head_tool(self.deps, direction=direction)
            await asyncio.sleep(1.0)
        await self._emotion_tool(self.deps, emotion="curious")

    def _listen_for_wake_phrase(self, samples: NDArray[np.float32]) -> None:
        # Spotting only runs while dormant or during the activation test, to save CPU.
        if not (self.is_dormant or self._wake_test_active):
            return
        if not self.wake_spotter.feed(samples):
            return
        if self._wake_test_active:
            self._wake_test_hits += 1
            logger.info("Wake phrase test: heard %d time(s)", self._wake_test_hits)
        else:
            self.request_wake("wake_phrase")

    async def run_wake_test(self, phrase: str, needed: int = 2, timeout_s: float = 20.0) -> int:
        """Listen for ``phrase`` for up to ``timeout_s``; returns how many times it was heard."""
        self.wake_spotter.set_phrase(phrase)
        deadline = time.monotonic() + 60.0
        while not self.wake_spotter.ready and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        self._wake_test_hits = 0
        self._wake_test_active = True
        try:
            deadline = time.monotonic() + timeout_s
            while self._wake_test_hits < needed and time.monotonic() < deadline:
                await asyncio.sleep(0.2)
        finally:
            self._wake_test_active = False
        return self._wake_test_hits

    async def change_voice(self, voice: str) -> str:
        """Switch the live voice and remember it for future sessions."""
        if self.stream is None:
            return "No stream is running."
        return await self.stream.change_voice(voice)

    async def complete_activation(self) -> None:
        """Mark the droid activated and push the full identity to the live session."""
        self.identity.activated = True
        self.save_identity()
        self.wake_spotter.set_phrase(self.identity.wake_phrase)
        if self.handler is not None:
            await self.handler.refresh_instructions()

    async def factory_reset(self) -> None:
        """Forget identity, faces and activation progress, then restart activation in a fresh session."""
        logger.warning("Droid factory reset requested")
        await self._run_hooks(self.on_factory_reset, "factory_reset")
        self.face_registry.clear()
        self.activation.reset()
        self.journal.take()
        self.identity = Identity()
        self.save_identity()
        self.wake_spotter.set_phrase("")
        if self.stream is not None:
            await self.stream.request_backend_restart("droid factory reset")

    # ---- memory ------------------------------------------------------------------

    def _import_legacy_facts(self) -> None:
        # One-time import of facts saved by the app's generic remember tool before the droid existed.
        if self.memory.get_state("legacy_imported"):
            return
        legacy = list_memory_facts(self.instance_path)
        for fact in legacy:
            self.memory.remember(fact.text)
        self.memory.set_state("legacy_imported", "1")
        if legacy:
            logger.info("Imported %d legacy memory facts", len(legacy))

    def memory_access_allowed(self) -> tuple[bool, str]:
        """Return whether memories may be shared now: not when only non-owners are in view."""
        in_view = self.people_in_view()
        if not in_view or self.owner_in_view():
            return True, ""
        return False, f"{self.identity.owner_name} is not in view ({', '.join(in_view)} is), so memories stay private"

    def _record_transcript(self, event: ConversationEvent) -> None:
        if event.kind == "user_transcript":
            if self.owner_in_view():
                self.memory.set_state(OWNER_LAST_SEEN_KEY, str(time.time()))
            self.journal.add("user", event.text)
        elif event.kind == "assistant_transcript":
            self.journal.add("assistant", event.text)
        else:
            return
        if len(self.journal.lines) >= CHECKPOINT_TURNS:
            self._spawn(self.checkpoint_session("checkpoint"))

    async def _checkpoint_on_dormant(self, reason: str) -> None:
        await self.checkpoint_session(reason)

    async def checkpoint_session(self, reason: str) -> None:
        """Summarise the transcript so far into an episode and durable facts."""
        if self._checkpointing:
            return
        started_at, lines = self.journal.take()
        if len(lines) < 2 or not self.identity.activated or not self.identity.memory_consent:
            return
        self._checkpointing = True
        try:
            known = [fact.text for fact in self.memory.facts(limit=60)]
            summary = await asyncio.to_thread(summarize_session, lines, self.identity, known)
            self.memory.add_episode(started_at, time.time(), summary.summary, summary.mood, summary.tags)
            for subject, text in summary.facts:
                self.memory.remember(text, subject)
            logger.info("Saved session episode (%s) with %d new facts", reason, len(summary.facts))
        finally:
            self._checkpointing = False

    async def _wipe_memory(self, reason: str) -> None:
        self.memory.wipe()

    def mood(self) -> str:
        """Return today's mood label for the status page."""
        return mood_for(self.memory, self.identity, time.time()).label

    async def _deliver_due_reminders(self) -> None:
        now = time.time()
        connected = self.handler is not None and self.handler._is_connected()
        for reminder in self.memory.due_reminders(now):
            # Give a session that is still connecting (e.g. just woken for this reminder) a minute to come up.
            if not connected and now - reminder.due_at < _REMINDER_SESSION_GRACE_S:
                continue
            self.memory.mark_delivered(reminder.id)
            await self.beep("alarm")
            spoken = await self.say_event(
                f"[SYSTEM EVENT] Reminder for {self.identity.owner_name}, due now: {reminder.text}. Deliver it in "
                "one short line."
            )
            if not self.owner_in_view() or not spoken:
                await send_notification(self.settings, reminder.text, title="Reminder", priority="high")

    def _once_per_day(self, key: str) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        if self.memory.get_state(key) == today:
            return False
        self.memory.set_state(key, today)
        return True

    async def _maybe_morning_briefing(self) -> None:
        now = datetime.now()
        if not (self.identity.activated and 5 <= now.hour < 12 and self.owner_in_view()):
            return
        if self.seconds_since_activity() < 20 or not self._once_per_day("briefing_day"):
            return
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        today_reminders = [r.text for r in self.memory.pending_reminders() if r.due_at < midnight + 86400]
        yesterday = self.memory.episodes(limit=1)
        await self.say_event(
            f"[SYSTEM EVENT] Morning briefing for {self.identity.owner_name}, first sighting today. In three short "
            "lines: greet them, give today's weather with your weather tool, then mention "
            + (f"today's reminders ({'; '.join(today_reminders)})" if today_reminders else "that nothing is scheduled")
            + (f", and one callback to the last session: {yesterday[0].summary}" if yesterday else "")
            + "."
        )

    async def _maybe_write_diary(self) -> None:
        now = datetime.now()
        if not self.identity.activated or not self.identity.memory_consent or now.hour < 23:
            return
        if not self._once_per_day("diary_day"):
            return
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        episodes = self.memory.episodes(limit=20, since=midnight)
        notes = [f"My mood today: {self.mood()}."]
        cat_visits = int(self.memory.get_state(f"cat_visits:{now:%Y-%m-%d}", "0"))
        if cat_visits:
            notes.append(f"The cat visited me {cat_visits} times.")
        entry = await asyncio.to_thread(write_diary_entry, self.identity, episodes, notes)
        self.memory.write_diary(now.strftime("%Y-%m-%d"), entry)
        logger.info("Diary entry written for %s", now.strftime("%Y-%m-%d"))

    # ---- senses and voice ------------------------------------------------------------

    def _apply_voice_fx(self, sample_rate: int, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        voice_filter = self._voice_filters.get(sample_rate)
        if voice_filter is None:
            voice_filter = self._voice_filters[sample_rate] = DroidVoiceFilter(sample_rate)
        return voice_filter.process(chunk)

    def _load_sound_watcher(self) -> None:
        try:
            self._sound_watcher = SoundWatcher(SoundClassifier().scores)
        except (OSError, ValueError) as e:
            logger.error("Sound classifier unavailable: %s", e)

    def _listen_for_sounds(self, samples: NDArray[np.float32]) -> None:
        if self.is_dormant:
            return
        if self._sound_watcher is None:
            if not self._sound_loading:
                self._sound_loading = True
                threading.Thread(target=self._load_sound_watcher, daemon=True, name="droid-sounds-load").start()
            return
        for event in self._sound_watcher.feed(samples):
            logger.info("Sound event: %s (%.2f)", event.kind, event.score)
            self.submit(self._react_to_sound(event))

    def _quiet_now(self) -> bool:
        return in_quiet_hours(self.identity.protocols.quiet_hours, datetime.now())

    async def _react_to_sound(self, event: SoundEvent) -> None:
        if event.kind == "music_stopped":
            await self._stop_dance_tool(self.deps)
            return
        if not self.identity.activated or self.is_dormant or self._quiet_now():
            return
        if event.kind == "meow":
            await self.react_to_cat(heard=True)
        elif event.kind == "music_started" and self.seconds_since_activity() >= _TALKING_RECENTLY_S:
            await self.beep("happy")
            await self._dance_tool(self.deps)

    async def _maybe_keep_dancing(self) -> None:
        watcher = self._sound_watcher
        if watcher is None or not watcher.music_playing or not self.identity.activated or self._quiet_now():
            return
        if self.seconds_since_activity() >= _TALKING_RECENTLY_S and self.deps.movement_manager.is_idle():
            await self._dance_tool(self.deps)

    async def react_to_cat(self, *, heard: bool) -> None:
        """Greet the cat: glance at it, trill, and say something kind (at most every ten minutes)."""
        now = time.monotonic()
        if now - self._last_cat_reaction < _CAT_REACTION_COOLDOWN_S or self._quiet_now():
            return
        self._last_cat_reaction = now
        visits_key = f"cat_visits:{datetime.now():%Y-%m-%d}"
        self.memory.set_state(visits_key, str(int(self.memory.get_state(visits_key, "0")) + 1))
        cat = self.presence.last_cat if self.presence is not None else None
        if not heard and cat is not None:
            await asyncio.to_thread(self.look_at_pixel, cat.u, cat.v)
        await self.beep("trill")
        await self.say_event(cat_prompt(self.identity, heard=heard))

    def look_at_pixel(self, u: int, v: int, duration_s: float = 1.5) -> None:
        """Glide the head towards a camera pixel through the movement queue."""
        robot = self.deps.reachy_mini
        try:
            target = robot.look_at_image(u, v, duration=duration_s, perform_movement=False)
            _, antennas = robot.get_current_joint_positions()
            self.deps.movement_manager.queue_move(
                GotoQueueMove(
                    target_head_pose=target,
                    start_head_pose=robot.get_current_head_pose(),
                    target_antennas=(antennas[0], antennas[1]),
                    start_antennas=(antennas[0], antennas[1]),
                    duration=duration_s,
                )
            )
        except Exception as e:
            logger.warning("Could not look at pixel (%d, %d): %s", u, v, e)
