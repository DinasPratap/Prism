# Final Advancement: GazeAssist multi-agent guidance

A LangGraph system with four cooperating agents and two memories, built on the GazeAssist design
(detect → risk → short spoken alerts). It speaks through ElevenLabs and stores house knowledge in Firebase.


## The agents

| # | Agent | How it works | Uses an LLM? |
|---|---|---|---|
| 1 | **Obstacle** | Sends a small downscaled frame to Gemini and gets back typed detections (label, left/right position, distance estimate), the room, and visible personal items. | Yes (vision) |
| 2 | **Path** | Free space in the left / ahead / right zones, a coarse hint ("More space on your left"), and a room-to-room route from permanent memory when you pass `--destination`. | No |
| 3 | **Risk** | Priority 1/2/3 from distance, time-to-collision and stairs rules. Same thresholds as the GazeAssist README. | No |
| 4 | **Output** | Builds short fixed-wording messages, throttles repeats (escalations always speak), then speaks via ElevenLabs. | No |

**Why only the first agent uses an LLM:** a language model is the right tool for *looking* at a scene, but not for
deciding "stop" in time. Risk and wording are plain code, so they are instant, predictable and testable.
Gemini's vision call is still the slowest step (often a second or more), so detection runs at about 1 Hz by default.

```
START ─ query? ─yes→ memory_read ──────────────────────────┐
   │ no                                                     ▼
   ▼                                   ┌→ path  (2) ──┐
obstacle (1) ──────────────────────────┼→ risk  (3) ──┼→ output (4) → ElevenLabs
                                       └→ memory ─────┘
```
`path`, `risk` and `memory` run in parallel.

## The two memories and the router

| Memory | Holds | Where | Size limit |
|---|---|---|---|
| **Recent buffer** | Rooms you recently went through, and where items were last seen or put ("keys, on the kitchen counter, 10 min ago") | RAM, snapshotted to Firebase every 5 min and on exit | 30 items, 20 rooms, 6 h expiry |
| **Permanent memory** | Directions between rooms, and usual places for items | Firebase Firestore (local JSON if no key) | Text only |

**The memory router** decides where each fact goes and which memory answers each question:

1. **Rules first** (instant, free): "I put my keys on the counter" → buffer. "From the bedroom to the kitchen, go straight 6 steps…" → permanent.
2. **Gemini only when rules can't decide**, using structured classification. Without a key it safely discards unclear notes.
3. **Promotion:** an item seen in the same place on 3 separate occasions, or a room-to-room move seen twice, moves from the buffer to permanent memory.
4. **Reads:** "where are my keys?" → buffer, then usual places. "how do I get to the kitchen?" → permanent route search, from your current room.

Memory writes run on a background thread, so they never delay an alert.

## Keeping memory use low

- Only the newest camera frame is kept (no queue), downscaled to 320 px and JPEG-compressed.
- The image is dropped from the graph state right after detection.
- No LangGraph checkpointer, so no history builds up.
- Both memories are bounded by count and age. The audio phrase cache holds 24 clips.
- Firebase is written only on promotion, on notes you teach it, and every 5 minutes for the buffer snapshot.

## Setup

```bash
cd "Final Advancement"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your keys
python -m gazeassist_agents doctor
```

Audio playback needs an MP3 player (`ffmpeg`'s `ffplay`, `mpg123`, or macOS `afplay`). Without ElevenLabs or a
player, the OS voice (`say`, `espeak-ng`) is used, and the console always prints every message.

**Firebase:** create a Firestore database, download a service-account key, save it as `firebase-key.json`, and set
`FIREBASE_CREDENTIALS` in `.env`. Data is stored under `users/{FIREBASE_USER_ID}/` as `edges`, `habits` and `meta`.

## Run

```bash
python -m gazeassist_agents selftest                                  # offline, no keys, 23 checks
python -m gazeassist_agents run --source mock --fast --seconds 40     # scripted walk, no keys needed
python -m gazeassist_agents run --source camera --interactive         # live: Gemini + ElevenLabs
python -m gazeassist_agents run --source camera --destination kitchen # also read out the stored route
```

With `--interactive` you can type while it runs:

```
I put my keys on the kitchen counter                      # note → buffer
From the bedroom to the kitchen: go straight 6 steps, then turn left   # note → permanent
where are my keys?                                        # question → answered aloud
```

## Known limits

- Gemini's distance estimates are rough and unvalidated. The prompt asks it to estimate closer when unsure.
- Latency is dominated by the network and the vision call. If vision fails twice in a row, the system says
  "Obstacle detection unavailable. Use your cane or ask for help."
- The prototype tracker matches objects by label and zone only. It is not a real multi-object tracker.
- Room recognition from a single frame can be wrong. A room must be seen twice in a row before it counts.
- Hasn't been run against live Gemini, ElevenLabs or Firebase from the build environment. Offline tests cover the logic;
  the live integrations are built to each library's documented API and need your keys to confirm.
- No voice input yet. Questions are typed. ElevenLabs speech-to-text is the natural next step.
- Audio plays on the machine running Python. For the Samsung glasses, pass `on_audio=` to `AudioOut` to forward the MP3
  bytes to your phone app.
