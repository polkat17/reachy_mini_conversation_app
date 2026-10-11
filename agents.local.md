# agents.local.md

Session notes for AI agents working on this fork, as the Reachy Mini SDK's `AGENTS.md` asks. Read this, then
`AGENTS.md` (this repository's rules), `DROID_BRIEF.md` and `DROID_ROADMAP.md`.

## Owner and robot

- Owner: Pasha. GitHub fork: `polkat17/reachy_mini_conversation_app`.
- Robot: **Reachy Mini Wireless**, so the app runs on the robot's onboard **CM4** (limited CPU and memory).
  RAM size not confirmed yet. Keep new on-robot models small and make heavy senses switchable.
- App flavour: Python app (extends the conversation app); the droid needs on-robot, always-on sensing.

## Project state

- The `droid` profile and `src/reachy_mini_conversation_app/droid/` are built through all roadmap phases.
- Not yet run on the physical robot; see the checklist in `DROID_ROADMAP.md`.

## Working environment notes

- Use Python 3.12 (CI's version); some realtime tests hang on 3.11. `uv sync --group dev`, then
  `ruff check . && ruff format . && mypy && pytest tests/`.
- Avatar SVGs are Git LFS files: run `git lfs pull` or three avatar tests fail.
- Smoke test against a simulated robot: `reachy-mini-daemon --mockup-sim --headless`, then run the app's
  `main.run()` with `ReachyMini(host="localhost", connection_mode="localhost_only", media_backend="no_media")`
  and `REACHY_MINI_CUSTOM_PROFILE=droid`. Cloud containers lack GStreamer's WebRTC plugin, so media must be off
  there; motion, the daemon API and the Hugging Face voice session still work. Ctrl+C does not stop the app in
  that media-less setup (the default profile behaves the same); kill the process instead.
