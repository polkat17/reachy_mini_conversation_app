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
