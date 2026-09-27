# Droid Companion for Reachy Mini — Build Brief

> For Claude Code. Drop this file in the root of the forked repo and start with:
> "Read DROID_BRIEF.md, explore the codebase, then propose a plan for Phase 1 before writing code."

## 1. Goal

Turn Reachy Mini (wireless) into a talking, Star Wars–style droid companion with:

1. **A persistent droid personality.** It has an original name, voice and character (fussy protocol droid or blunt deadpan droid). It must not copy existing characters or voices.
2. **Long-term memory.** It remembers facts, people and past conversations across sessions, and the owner stays in control of what is kept.
3. **Expressive body language.** Every reply is paired with head and antenna emotes.
4. **Later: game commentary.** It reacts to the owner playing console games (see Phase 4).

## 2. Base codebase

This repo is a fork of `pollen-robotics/reachy_mini_conversation_app`. **Extend it rather than rewrite it.** It already provides:

- A realtime voice conversation loop (Hugging Face realtime backend, with OpenAI realtime also supported in forks)
- A tool system the LLM can call (emotions, head pose, head tracking, camera capture, idle, sleep)
- Personality profiles (name, instructions, greeting) and an optional web UI (`--ui`, port 7860)
- A layered motion system: queued primary moves, a neutral idle pose, and speech-reactive wobble
- The emotions library dataset (`pollen-robotics/reachy-mini-emotions-library`)

**First task:** map the repo. Find where profiles, tools, the session lifecycle and the audio output live. Summarise what you find before changing anything, and confirm my assumptions below against the real code.

## 3. Architecture

```
            ┌──────────────── Reachy Mini (Pi 5, wireless) ───────────────┐
            │  mic ─┐          camera ─┐        speaker ◄─┐   motors ◄─┐  │
            └───────┼──────────────────┼────────────────────┼────────────┼──┘
                    │ daemon (SDK)     │                    │            │
┌───────────────────▼──────────────────▼────────────────────┼────────────┼──┐
│  PC / host: conversation app (this fork)                  │            │  │
│                                                           │            │  │
│  Realtime voice backend ── audio out ──► [Voice FX] ──────┘            │  │
│        │  ▲                                                            │  │
│        │  │ instructions = Persona + Core memory + recalled memories   │  │
│        ▼  │                                                            │  │
│   Tool calls ──► Motion tools (existing) ──────────────────────────────┘  │
│        │                                                                  │
│        └──────► Memory tools (NEW) ──► Memory Service (NEW)               │
│                                           ├─ core.yaml   (identity/facts) │
│                                           ├─ memory.db   (SQLite)         │
│                                           │    episodes + facts + vectors │
│                                           └─ local embedding model        │
│                                                                           │
│  Session end hook (NEW) ──► summarise transcript ──► new episode          │
└───────────────────────────────────────────────────────────────────────────┘
```

## 4. Components to build

### 4.1 Droid persona profile
- Create a new profile `droid` with an original name (placeholder: `K3-L0`; the owner will rename it), full instructions and a greeting.
- Character rules: speak in character, keep replies short (1 to 3 sentences, since it is spoken aloud), use dry humour, occasionally refer to itself as a droid, and never claim to be a real Star Wars character.
- Instruct it to **call an emote tool with every reply**. Build a small mapping of named emotions to existing emotion or move assets: `curious`, `startled`, `sad`, `happy`, `smug`, `worried`, `thinking`.
- Owner name: Pasha.

### 4.2 Memory service (`memory/` package)
- Storage is a local SQLite file at `~/.droid/memory.db`. Keep it out of the repo and add it to `.gitignore`.
- Tables:
  - `facts(id, subject, text, created_at, updated_at)`: stable facts ("Pasha's partner is called X", "Pasha supports Y").
  - `episodes(id, started_at, ended_at, summary, mood, tags)`: one row per conversation session.
  - `embeddings(ref_type, ref_id, vector)`: use `sqlite-vec` or a Chroma sidecar. Pick whichever is simpler here and justify the choice.
- Embeddings run locally with a small sentence-transformers model (for example `all-MiniLM-L6-v2`), so there is no extra API cost.
- `core.yaml` is hand-editable and holds the droid's identity plus the most important owner facts. It is always injected.
- **Never store raw audio or video.** Store text summaries only.

### 4.3 Memory tools (register with the existing tool system)
- `remember(text, subject?)`: save a fact when the owner says "remember that…" or when something clearly durable comes up.
- `recall(query, k=5)`: semantic search over facts and episodes. The LLM calls this when a past event seems relevant.
- `forget(query)`: find matches, have the droid read them back and confirm, then delete them.
- `list_memories(subject?)`: supports "what do you remember about me?"

### 4.4 Context injection at session start
- Build the instructions from: persona, then `core.yaml`, then the top facts, then summaries of the last 3 episodes.
- Keep the injected memory block under ~1,500 tokens. Everything else is reached through `recall`.

### 4.5 Session end hook
- Keep a rolling text transcript during the session (use the backend's transcription events if they are available; confirm this in the code).
- When the session ends (sleep, exit or idle timeout), send one LLM call to summarise it into an episode (summary, mood, tags) and extract any new facts. Deduplicate against existing facts and update them rather than adding duplicates.

### 4.6 Voice FX (Phase 2)
- Use a realtime voice from the backend catalogue, then add a light "droid" filter on the output stream: subtle ring modulation, a band-pass filter and a slight metallic reverb.
- Latency budget is **under 20 ms** added. Include a config flag to turn it off. If the audio path makes this hard, report back instead of hacking around it.

## 5. Phases

1. **Phase 1: Persona + emotes.** Droid profile, emotion mapping, runs end to end. Done when I can hold a conversation and it emotes on every reply.
2. **Phase 2: Memory.** Service, tools, injection and session hook. Done when it recalls something from yesterday's session without being prompted, and `forget` works.
3. **Phase 3: Voice FX + polish.** Filter, wake-word/sleep flow, face tracking on by default.
4. **Phase 4: Game commentary (later, scoping only for now).** Two inputs: its own camera watching the owner, and a capture-card or screen feed of the console game. Only design the interface for now (a `GameFeed` source that produces periodic frame descriptions), and don't implement it yet.

## 6. Constraints and conventions

- Python and the repo's existing tooling and style. Add minimal new dependencies and list each one with a reason.
- Put all secrets in `.env`. Never commit keys.
- Configuration lives in one place (`droid.yaml`): persona name, voice, FX on/off, memory paths, token budgets.
- Write tests for the memory service (unit tests using a temporary database). Test the robot-facing code with the `--no-camera` path and mocks where possible.
- Keep changes to upstream files small and clearly marked, so upstream updates can still be merged.
- Before each phase: give a short plan. After each phase: a summary of what changed and how to run it.

## 7. Open questions for Claude Code to raise with me
- Which realtime backend to use as the default (HF or OpenAI), trading off cost, latency and voice quality.
- Whether the backend exposes text transcripts reliably enough for the session summary.
- The best hook point for the output-audio filter.
