"""The droid runtime: a light layer over the conversation that gives it a persistent identity and memory.

It only listens to conversation events. It plays an emotion that fits what the droid says (the model itself only
talks), keeps a transcript of the current conversation and, once the
conversation goes quiet (or gets long, or the app stops), summarises it into an episode and durable facts
with one background LLM call. Nothing here polls the camera, the microphone or the motors.
"""

import time
import asyncio
import logging
import threading
from typing import Any
from pathlib import Path
from datetime import datetime

from k3_droid.config import config
from k3_droid.daemon_api import DaemonApiError, daemon_request
from k3_droid.app_lifecycle import power_up_quietly
from k3_droid.droid.context import open_memory
from k3_droid.droid.journal import CHECKPOINT_TURNS, SessionJournal, summarize_session
from k3_droid.droid.updater import (
    latest_revision,
    load_update_state,
    save_update_state,
    installed_revision,
    write_health_beacon,
    launch_update_helper,
    take_boot_dormant_flag,
)
from k3_droid.droid.identity import Identity, load_identity, save_identity, droid_data_dir
from k3_droid.droid.settings import DroidSettings
from k3_droid.droid.autostart import current_app_name, ensure_autostart
from k3_droid.droid.day_review import fetch_day_review
from k3_droid.tools.core_tools import ToolDependencies
from k3_droid.tools.play_emotion import PlayEmotion
from k3_droid.droid.body_language import emotion_for_reply, requested_emotion
from k3_droid.conversation_handler import ConversationEvent, ConversationHandler


logger = logging.getLogger(__name__)

_QUIET_BEFORE_SUMMARY_S = 5 * 60.0
_SUMMARY_POLL_S = 30.0
_DAY_REVIEW_RECHECK_S = 600.0
_EMOTION_GAP_S = 6.0  # at most one unrequested emotion this often, so the head is not always busy


class DroidRuntime:
    """Identity, memory and session summaries for the droid profile."""

    def __init__(self, deps: ToolDependencies, instance_path: str | Path | None) -> None:
        """Load the identity and open the memory; nothing starts until ``start``."""
        self.deps = deps
        self.instance_path = instance_path
        self.settings = DroidSettings.from_env()
        self.identity: Identity = load_identity(instance_path)
        self.data_dir = droid_data_dir(instance_path)
        self.memory = open_memory(instance_path)
        self.journal = SessionJournal()
        self.handler: ConversationHandler | None = None
        self._stop = threading.Event()
        self._summary_lock = threading.Lock()
        self._day_review: dict[str, Any] | None = None
        self._day_review_checked_at: float | None = None
        self.day_review_note = "not checked yet"
        self.app_name: str | None = None
        self._emotion_tool = PlayEmotion()
        self._emotion_lock = threading.Lock()
        self._requested_emotion: str | None = None
        self._last_emotion_at = float("-inf")

    # ---- lifecycle --------------------------------------------------------------

    def attach(self, handler: ConversationHandler) -> None:
        """Listen to ``handler``; called for every handler the app builds."""
        self.handler = handler
        handler.set_conversation_listener(self._on_event)

    def start(self) -> None:
        """Report a healthy boot, register as the startup app and start the quiet-conversation summariser."""
        # An update restarted this app and waits for this beacon; without it, it rolls the update back.
        write_health_beacon(self.data_dir)
        take_boot_dormant_flag(self.data_dir)  # left by older versions; the droid no longer sleeps on its own
        if self.settings.autostart:
            threading.Thread(target=self._register_app, daemon=True, name="droid-autostart").start()
        threading.Thread(target=self._summarise_when_quiet, daemon=True, name="droid-summaries").start()

    def stop(self) -> None:
        """Save what was said in the current conversation, then close the memory."""
        self._stop.set()
        self.save_session("app stopping")
        self.memory.close()

    def _register_app(self) -> None:
        self.app_name = current_app_name(self.deps.reachy_mini)
        if self.app_name is not None:
            ensure_autostart(self.deps.reachy_mini, self.app_name)

    # ---- conversation events ---------------------------------------------------------

    def _on_event(self, event: ConversationEvent) -> None:
        if event.kind not in ("user_transcript", "assistant_transcript"):
            return
        self.journal.add("user" if event.kind == "user_transcript" else "assistant", event.text)
        if len(self.journal.lines) >= CHECKPOINT_TURNS:
            threading.Thread(target=self.save_session, args=("long conversation",), daemon=True).start()
        if event.kind == "user_transcript":
            self._requested_emotion = requested_emotion(event.text)
        else:
            self._react_to_reply(event.text)

    def _react_to_reply(self, reply: str) -> None:
        """Play the emotion the owner asked for, or one that fits the reply, while the reply is being spoken."""
        emotion, self._requested_emotion = self._requested_emotion, None
        now = time.monotonic()
        if emotion is None:
            emotion = emotion_for_reply(reply)
            if emotion is None or now - self._last_emotion_at < _EMOTION_GAP_S:
                return
        self._last_emotion_at = now
        # The first play downloads the emotions library, so keep it off the conversation's event loop.
        threading.Thread(target=self._play_emotion, args=(emotion,), daemon=True, name="droid-emotion").start()

    def _play_emotion(self, emotion: str) -> None:
        with self._emotion_lock:
            result = asyncio.run(self._emotion_tool(self.deps, emotion=emotion))
        if "error" in result:
            logger.warning("Could not play emotion %s: %s", emotion, result["error"])

    # ---- memory -----------------------------------------------------------------------

    def _summarise_when_quiet(self) -> None:
        while not self._stop.wait(_SUMMARY_POLL_S):
            handler = self.handler
            quiet_s = float("inf") if handler is None else time.monotonic() - handler.last_activity_time
            if self.journal.lines and quiet_s >= _QUIET_BEFORE_SUMMARY_S:
                self.save_session("conversation went quiet")

    def save_session(self, reason: str) -> None:
        """Summarise the transcript so far into an episode and durable facts (one LLM call)."""
        if not self._summary_lock.acquire(blocking=False):
            return
        try:
            started_at, lines = self.journal.take()
            if len(lines) < 2 or not self.identity.memory_consent:
                return
            known = [fact.text for fact in self.memory.facts(limit=60)]
            summary = summarize_session(lines, self.identity, known)
            self.memory.add_episode(started_at, time.time(), summary.summary, summary.mood, summary.tags)
            for subject, text in summary.facts:
                self.memory.remember(text, subject)
            logger.info("Saved conversation (%s) with %d new facts", reason, len(summary.facts))
        except Exception as e:
            logger.warning("Could not save the conversation (%s): %s", reason, e)
        finally:
            self._summary_lock.release()

    def owner_access_allowed(self) -> tuple[bool, str]:
        """Return whether personal data may be shared; the droid has no face recognition, so always yes."""
        return True, ""

    def save_identity(self) -> None:
        """Persist the identity (name, owner, language)."""
        save_identity(self.instance_path, self.identity)

    # ---- day review -------------------------------------------------------------------

    def day_review(self) -> dict[str, Any] | None:
        """Return today's review written by the Claude routine, re-checking the dataset at most every 10 minutes."""
        dataset = DroidSettings.from_env().day_review_dataset
        if not dataset:
            self.day_review_note = "not set up (DROID_DAY_REVIEW_DATASET)"
            return None
        now = time.monotonic()
        today = datetime.now().date()
        if self._day_review_checked_at is None or now - self._day_review_checked_at >= _DAY_REVIEW_RECHECK_S:
            self._day_review, self.day_review_note = fetch_day_review(dataset, config.HF_TOKEN, today)
            self._day_review_checked_at = now
        review = self._day_review
        return review if review is not None and review.get("date") == today.isoformat() else None

    # ---- control page actions -------------------------------------------------------------

    def restore_motors(self) -> str:
        """Clear a stuck daemon move, turn the motors on and restart the control loop."""
        robot = self.deps.reachy_mini
        # The daemon ignores every app target while one of its own moves runs, so a move that never ended freezes the
        # robot; stop whatever it still lists as running.
        try:
            for move in daemon_request(robot, "/api/move/running") or []:
                daemon_request(robot, "/api/move/stop", method="POST", payload=move)
        except DaemonApiError as e:
            logger.warning("Could not clear running daemon moves: %s", e)
        self.deps.movement_manager.stop(reset_to_neutral=False)
        try:
            robot.enable_wobbling()
            restored = power_up_quietly(robot, logger)
        except Exception as e:  # never leave the control loop stopped behind a failed call
            logger.error("Restoring the motors failed: %s", e)
            restored = False
        finally:
            self.deps.movement_manager.start()
        if not restored:
            return "Could not switch the motors on; use Restart app."
        return "Motors restored: torque on, head back to neutral, movement restarted."

    def restart_app(self) -> str:
        """Ask the daemon to restart this app (rebuilds audio and motion from scratch)."""
        threading.Thread(target=self._request_restart, daemon=True, name="droid-restart").start()
        return "Restarting the app; the page comes back in about half a minute."

    def _request_restart(self) -> None:
        time.sleep(0.5)  # let the HTTP reply reach the browser first
        try:
            daemon_request(self.deps.reachy_mini, "/api/apps/restart-current-app", method="POST")
        except DaemonApiError as e:
            logger.error("Could not restart the app: %s", e)

    def check_for_update(self) -> str:
        """Install the update Space's latest revision if it differs from the installed one; returns what happened."""
        space = self.settings.update_space
        app_name = self.app_name or current_app_name(self.deps.reachy_mini)
        if not space or app_name is None:
            return "Updates are not set up (DROID_UPDATE_SPACE, and the app must be started by the robot)."
        latest = latest_revision(space, config.HF_TOKEN)
        if latest is None:
            return "Could not reach the update Space."
        state = load_update_state(self.data_dir)
        # Trust pip's record of the installed snapshot: the saved state goes stale after a daemon (re)install.
        installed = installed_revision(app_name, space) or state.installed_revision
        state.installed_revision, state.last_checked_at = installed, time.time()
        save_update_state(self.data_dir, state)
        if installed and latest == installed:
            return "Already up to date."
        client = self.deps.reachy_mini.client
        launch_update_helper(
            {
                "space": space,
                "revision": latest,
                "previous": installed,
                "app": app_name,
                "daemon_url": f"http://{client.host}:{client.port}",
                "data_dir": self.data_dir,
                "hf_token": config.HF_TOKEN or "",
                "boot_dormant": "0",
            }
        )
        return f"Installing {latest[:8]}; the droid restarts in a moment."
