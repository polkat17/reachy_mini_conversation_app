# Droid Companion Roadmap

Companion to `DROID_BRIEF.md`. It records the decisions agreed with Pasha and the order the droid is built in.
Everything droid-specific lives in `src/reachy_mini_conversation_app/droid/` and `profiles/droid/`, so upstream
changes to the conversation app still merge cleanly. Upstream files get small, clearly marked hooks only.

## Decisions

| Topic | Decision |
|---|---|
| Runs on | The robot (Pi 5). Models stay light and run on ONNX Runtime, which the SDK already ships. |
| Voice AI | Hugging Face realtime backend (the only backend in this fork). |
| Owner | Pasha, preset. The droid still asks for its own name during activation. |
| Strangers | Talks freely, but never shares Pasha's memories. |
| Wake-up | Seeing Pasha's face, hearing its wake phrase (chosen during activation), or the web UI button. |
| Autostart | The daemon's built-in startup-app setting (`PUT /api/apps/startup-app`). No SSH needed. |
| Self-upgrades | The droid files a GitHub issue; a scheduled Claude Code session opens a PR; Pasha merges; the robot installs `main` while dormant and rolls back if the new version fails its health check. |
| Settings | `.env` (documented in `.env.example`). |
| Storage | The app's instance data folder: memory database, identity file, face embeddings. Text only, never raw audio or video. |

## Phases

All seven phases are built. Everything is unit-tested with mocks; items marked in the checklist below still
need a first run on the robot.

1. **Persona and emotes**: `droid` profile, droid emotion names, an emote on every reply, beep language.
2. **Always on**: dormant/wake instead of exiting, autostart, idle life, phone notifications (ntfy).
3. **Activation and presence**: first-boot activation protocol, face enrolment and recognition, wake phrase,
   guest registry, welcome home / goodbye, break nudges.
4. **Memory**: SQLite memory with semantic recall, confirmed forget, episodes, droid diary, moods over days,
   morning briefing, reminders and timers.
5. **Senses and voice**: sound classifier (music and meows), cat detection and cat talk, show and tell,
   droid voice filter.
6. **Growth and home**: self-development pipeline, upgrade suggestions, Home Assistant smart home.
7. **Game commentary**: `GameFeed` interface only.

## Activation protocol

Runs on the first power-on, resumes if interrupted, and every stage can be redone by voice later.

0. Boot self-test (head scan, antenna twitch, beeps).
1. Owner registration: Pasha faces the camera; the droid enrols his face and confirms it recognises him.
2. Designation: keep `K3-L0` or assign a new name.
3. Wake phrase: 2 to 4 words, distinct from the name; tested three times before it is accepted.
4. Vocal module: pick from spoken voice samples.
5. Personality calibration: humour, talkativeness, language.
6. Primary directives: interests, topics to follow, topics to avoid.
7. Interaction protocols: greetings, unprompted remarks, quiet hours, face following.
8. Household registry: people and pets (including the cat), and how to treat guests.
9. Memory consent.
10. Upgrade protocol briefing.
11. Commit: the droid reads the profile back and saves it after confirmation.

## Setup

1. **Install the fork on the robot.** Create a Hugging Face Space (for example `polkat17/k3-droid`), then in the
   GitHub fork set the repository variable `DROID_HF_SPACE` to its id and the secret `HF_TOKEN` to a token with
   write access. The `Droid - sync main to Hugging Face Space` workflow mirrors `main` on every push. Install
   the Space from the robot dashboard like any app.
2. **Select the droid.** In the app's settings page choose the `droid` personality (or set
   `REACHY_MINI_CUSTOM_PROFILE=droid`).
3. **Optional settings** in the app's `.env` (see `.env.example`):
   - `DROID_NTFY_TOPIC`: a long, private topic name; subscribe to it in the ntfy phone app.
   - `DROID_GITHUB_REPO` and `DROID_GITHUB_TOKEN`: a fine-grained token limited to this fork with only
     "Issues: read and write".
   - `DROID_UPDATE_SPACE`: the Space from step 1, so merged upgrades install at night.
   - `DROID_HOME_ASSISTANT_URL` and `DROID_HOME_ASSISTANT_TOKEN`: a Home Assistant long-lived token.
   - `HF_TOKEN`: used for session summaries and the diary (`DROID_SUMMARY_MODEL`).
4. **Start it once from the dashboard.** It registers itself as the startup app. From then on: power on, flick
   an antenna, and the droid boots. The first boot runs the activation protocol.
5. **Control page:** `http://<robot>:7860/droid`.

## Scheduled upgrade sessions (Claude Code)

A Claude Code routine on your plan implements upgrade requests. Suggested schedule: nightly. Prompt:

> In `polkat17/reachy_mini_conversation_app`, list open issues labelled `droid-upgrade` that have no linked
> pull request. Take the oldest one. Read `AGENTS.md`, `DROID_ROADMAP.md` and the issue, implement it on a new
> branch `droid-upgrade/<issue-number>-<short-name>` following the repository's conventions (a new tool in
> `src/reachy_mini_conversation_app/tools/` enabled in `profiles/droid/profile.md`, or a change inside
> `src/reachy_mini_conversation_app/droid/`), add tests, run
> `ruff check . && ruff format --check . && mypy && pytest tests/`, and open a pull request that says
> "Closes #<number>" and lists any setup steps. Never merge. If the request is unsafe (unlocking doors, spending
> money, sharing the owner's data) or unclear, comment on the issue instead of implementing it. Stop after one
> issue.

## First run on the robot: checklist

These parts depend on real hardware or services and could only be tested with mocks here:

- [ ] Dormant and wake: sleep pose, session closes, wake-up move, and the movement loop restarting smoothly.
- [ ] Autostart: the app is listed as the daemon's startup app, and flicking an antenna after power-on boots it.
- [ ] Face enrolment and recognition in your room's lighting (threshold `MATCH_THRESHOLD` in `droid/faces.py`).
- [ ] Wake phrase through the robot's microphone (verified here only on synthesised speech).
- [ ] Loudness wake threshold (`LoudnessTrigger` in `droid/audio.py`) in your room.
- [ ] Music and meow detection through the robot's microphone; cat detection through its camera.
- [ ] Voice filter taste (parameters in `droid/voice_fx.py`).
- [ ] The update helper surviving the app being stopped, and pip installing into the apps environment.
- [ ] Session summaries with your `HF_TOKEN` and the chosen `DROID_SUMMARY_MODEL`.

## Known limits

- Face recognition is for convenience, not security; a photo can fool it.
- Memories are injected when a session starts, so a stranger who wakes the droid by voice (before the camera
  sees them) is covered only by the persona's guest rule until they are in view.
- Head following uses the SDK's face tracking; the cat is glanced at, not followed.
