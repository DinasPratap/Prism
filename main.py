"""
main.py  --  THE ENTRY POINT of the whole system.

Start it with:   python -m gazeassist_agents  {run | doctor | selftest}
(`__main__.py` simply imports and calls `main()` from this file.)

HOW THE PROJECT FITS TOGETHER (read in this order if you are new):
  1. main.py            (this file)  command-line interface + the main loop that feeds frames to the graph
  2. graph.py           wires the 4 agents + memory router together with LangGraph
  3. agents/obstacle.py   Agent 1: looks at a frame (Gemini vision) and lists obstacles, room, items
  4. agents/path.py       Agent 2: free space left/ahead/right + a route from permanent memory
  5. agents/risk.py       Agent 3: decides priority 1 (critical) / 2 (warning) / 3 (caution)
  6. agents/output.py     Agent 4: writes the short sentence to say
  7. audio.py           speaks that sentence with ElevenLabs (falls back to the OS voice)
  8. memory/            the two memories (RAM buffer + Firebase) and the router that picks between them

THE LOOP IN ONE PICTURE (what `run` does, about once per second):
    camera frame -> shrink + JPEG -> graph.invoke() -> [obstacle -> path/risk/memory -> output] -> speech
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

# Everything below is a piece of the system, imported from the sibling modules.
from .agents.obstacle import GeminiVision, MockVision   # the two "eyes": real Gemini, or a scripted fake
from .audio import AudioOut                             # speech output (ElevenLabs + fallback)
from .config import get_llm, load_settings              # reads .env; builds the shared Gemini client
from .frames import NewestFrameSource, encode_frame     # camera reader + frame shrinker
from .graph import build_graph                          # the LangGraph that connects the agents
from .memory.buffer import RecentBuffer                 # memory #1: short-term RAM buffer
from .memory.permanent import LocalJsonStore, make_store  # memory #2: Firebase (or local JSON)
from .memory.router import MemoryRouter                 # decides which memory gets/answers what

# How often (seconds) the short-term buffer is saved to the database so a restart doesn't lose it.
SAVE_BUFFER_EVERY_S = 300


# ============================================================================
# build(): create every component and connect them. Used by `run` and `selftest`.
# ============================================================================
def build(settings, vision, console_only: bool, store=None):
    """Assemble the full system.

    settings      -- values from .env (API keys, model names, ...)
    vision        -- the "eyes": GeminiVision (real) or MockVision (scripted, no keys needed)
    console_only  -- True = print messages but do not make any sound (used by tests)
    store         -- permanent-memory database; if None we pick Firebase, or a local JSON file

    Returns (graph, router, audio, store, buffer) so callers can use any of them.
    """
    # Permanent memory: Firebase if credentials exist, otherwise a local JSON file.
    store = store or make_store(settings)

    # Short-term memory lives in RAM. We try to restore what it remembered last time.
    buf = RecentBuffer()
    try:
        buf.load(store.load_buffer())
    except Exception:
        pass  # first run, or the database is unreachable: just start empty

    # The router sits between the two memories (and uses Gemini only when its rules can't decide).
    router = MemoryRouter(buf, store, get_llm(settings))

    # The speaker. console_only=True means "no sound", handy for tests and demos.
    audio = AudioOut(settings, console_only=console_only)

    # Finally connect agents 1-4 and the router into one LangGraph pipeline.
    return build_graph(vision, router, audio), router, audio, store, buf


# ============================================================================
# `run`: the real thing. Capture a frame, push it through the agents, repeat.
# ============================================================================
def cmd_run(a) -> int:
    s = load_settings()

    # "mock" means a scripted fake walk: no camera, no Gemini key needed. Great for demos.
    mock = a.source == "mock"

    # For a real camera/video we NEED the Gemini key, so stop early with a clear message.
    if not mock and not s.gemini_api_key:
        print("GEMINI_API_KEY is missing. Add it to .env, or use --source mock.", file=sys.stderr)
        return 2

    # Pick the "eyes": the scripted fake or the real Gemini vision model.
    vision = MockVision() if mock else GeminiVision(get_llm(s))

    # Build the whole system (graph + memories + speaker).
    graph, router, audio, store, buf = build(s, vision, a.console_audio)

    # Start the camera reader (a background thread that keeps ONLY the newest frame).
    src = None if mock else NewestFrameSource(a.camera, a.video)

    # ---- optional: let the user type notes/questions while the system runs ----
    if a.interactive:
        def stdin_loop():
            print("Type a question ending in '?' (e.g. where are my keys?) or a note (I put my keys on the shelf).")
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                if line.endswith("?"):
                    # A question: run the graph's "query" path (memory router -> output agent -> speech).
                    graph.invoke({"query": line, "t": time.time()})
                else:
                    # A note: the router decides whether it is a short-term or permanent fact, then we confirm aloud.
                    audio.speak(router.remember(line), 3)
        threading.Thread(target=stdin_loop, daemon=True).start()

    period = 1.0 / a.hz   # seconds between frames (default 1 Hz, i.e. one frame per second)
    start = time.time()   # wall-clock start, used to stop after --seconds
    sim_t = 0.0           # simulated clock, only used by `--source mock --fast`
    last_save = time.time()
    print(f"Running ({a.source}, {a.hz} Hz). Ctrl+C to stop. Not a mobility aid: keep your cane.")

    try:
        while True:
            loop_start = time.time()
            fast_mock = mock and a.fast   # "fast" mock skips waiting and just simulates time passing

            # Stop after --seconds (0 means "run until Ctrl+C").
            if a.seconds and ((sim_t if fast_mock else loop_start - start) >= a.seconds):
                break

            # Every agent uses this timestamp (epoch seconds). Simulated in fast-mock, real otherwise.
            t = start + sim_t if fast_mock else loop_start

            # The graph's input. `destination` (optional) asks Agent 2 for a stored route to a room.
            state = {"t": t, "destination": a.destination}

            if not mock:
                if src.finished:          # a video file ended
                    break
                frame = src.get()         # newest frame (may be None for a moment at startup)
                # Shrink + compress so we send a tiny image to Gemini (less memory, bandwidth and cost).
                state["frame_b64"] = (
                    encode_frame(frame, s.frame_width, s.jpeg_quality) if frame is not None else None
                )

            # >>> THE MAIN CALL <<<  runs Agent 1 -> (Agents 2, 3 + memory) -> Agent 4 -> speech.
            graph.invoke(state)

            sim_t += period

            # Periodically save the short-term buffer so a crash or restart doesn't lose it.
            if time.time() - last_save > SAVE_BUFFER_EVERY_S:
                store.save_buffer(buf.dump())
                last_save = time.time()

            # Wait out the rest of this frame's time slot (not needed in fast-mock).
            if not fast_mock:
                time.sleep(max(0.0, period - (time.time() - loop_start)))
    except KeyboardInterrupt:
        pass  # Ctrl+C is the normal way to stop
    finally:
        # Tidy shutdown: let queued memory writes and speech finish, save the buffer, release the camera.
        router.flush()
        audio.wait()
        try:
            store.save_buffer(buf.dump())
        except Exception as exc:
            print(f"[store] could not save buffer: {type(exc).__name__}", file=sys.stderr)
        if src:
            src.close()
    return 0


# ============================================================================
# `doctor`: quick health check. Prints ok/-- per item and NEVER prints secret values.
# ============================================================================
def cmd_doctor(_a) -> int:
    s = load_settings()
    mark = lambda ok: "ok " if ok else "-- "
    print(mark(s.gemini_api_key), "GEMINI_API_KEY (vision + memory router)")
    print(mark(s.eleven_api_key), "ELEVENLABS_API_KEY (voice; falls back to the OS voice without it)")
    print(mark(s.firebase_credentials and Path(s.firebase_credentials).exists()),
          "FIREBASE_CREDENTIALS (otherwise a local JSON file is used)")
    # Check each optional library is installed.
    for mod in ("langgraph", "langchain_google_genai", "elevenlabs", "firebase_admin", "cv2"):
        try:
            __import__(mod)
            print(mark(True), "import", mod)
        except ImportError:
            print(mark(False), "import", mod, "(not installed)")
    print(mark(any(shutil.which(p) for p in ("afplay", "ffplay", "mpg123"))), "MP3 player (afplay / ffplay / mpg123)")
    return 0


# ============================================================================
# `selftest`: 23 offline checks. No keys, no camera, no internet needed.
# ============================================================================
def cmd_selftest(_a) -> int:
    from .agents.risk import priority

    results: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        """Record and print one PASS/FAIL line."""
        results.append((name, bool(ok)))
        print(("PASS " if ok else "FAIL ") + name)

    # ---- Agent 3 (risk rules): are the distance/time thresholds right? ----
    check("0.8 m is critical", priority("chair", "ahead", 0.8, None) == 1)
    check("1.7 m is warning", priority("person", "left", 1.7, None) == 2)
    check("2.5 m is caution", priority("chair", "ahead", 2.5, None) == 3)
    check("5 m is ignored", priority("chair", "ahead", 5.0, None) == 0)
    check("stairs at 1.5 m is critical", priority("stairs", "ahead", 1.5, None) == 1)
    check("unknown-distance stairs ahead is a warning", priority("stairs", "ahead", None, None) == 2)
    check("TTC 1.2 s is critical", priority("car", "ahead", 4.0, 1.2) == 1)

    # A throwaway folder + local store so tests never touch your real data or Firebase.
    tmp = Path(tempfile.mkdtemp())
    settings = load_settings()
    store = LocalJsonStore(tmp)
    buf = RecentBuffer(max_items=5)
    router = MemoryRouter(buf, store, llm=None)   # llm=None: tests must not need Gemini

    # ---- Memory 1: the buffer must never grow without limit ----
    for i in range(50):
        buf.see_item(f"thing{i}", "in the hall", 1000.0 + i)
    check("buffer stays bounded", len(buf.items) <= 5)

    # ---- Memory router: does it send each note to the right memory? ----
    d1 = router.decide("I put my keys on the kitchen counter")
    check("router sends item location to the buffer", d1.store == "buffer" and d1.item == "keys")
    d2 = router.decide("From the bedroom to the kitchen: go straight 6 steps, then turn left at the hallway")
    check("router sends directions to permanent memory", d2.store == "permanent" and d2.to_room == "kitchen")
    check("router discards chatter when no LLM", router.decide("nice weather today").store == "discard")

    # ---- Router reads: do questions get answered from the right memory? ----
    t0 = 1_000_000.0
    router.remember("I put my keys on the kitchen counter", t0)
    ans = router.answer("where are my keys?", t0 + 600)   # asked 10 minutes later
    check("buffer answers item questions", "kitchen counter" in ans and "10 minutes ago" in ans)
    router.remember("From the bedroom to the kitchen: go straight 6 steps, then turn left", t0)
    buf.visit("bedroom", t0)                               # pretend we are standing in the bedroom
    check("permanent memory answers route questions", "6 steps" in router.answer("how do I get to the kitchen?", t0))
    router.remember("From the kitchen to the garden: go through the back door", t0)
    check("route planning chains rooms", router.route_to("bedroom", "garden") is not None)

    # ---- Promotion: the same item in the same place on 3 separate occasions becomes permanent ----
    r2 = MemoryRouter(RecentBuffer(), LocalJsonStore(Path(tempfile.mkdtemp())), llm=None)
    for i in range(3):
        r2._observe("kitchen", ["wallet"], t0 + i * 700)       # 700 s apart = separate occasions
        r2._observe("kitchen", ["wallet"], t0 + i * 700 + 1)
    check("repeated sightings are promoted to permanent memory", "wallet" in r2.store.habits())

    # ---- The whole graph on the scripted 30 s walk (run for 45 s to cover a repeated room move) ----
    vision = MockVision()
    s_tmp = LocalJsonStore(Path(tempfile.mkdtemp()))
    graph, router3, audio, _store, buf3 = build(settings, vision, True, s_tmp)
    base = 2_000_000.0
    for step in range(45):
        graph.invoke({"t": base + step})   # one simulated second per call
    router3.flush()                         # wait for the background memory writes to finish
    said = [t for _, t in audio.history]    # everything the system "spoke"
    check("graph speaks a critical stop", any("Stop." in t for t in said))
    check("graph warns about stairs", any("Stairs" in t for t in said))
    check("graph announces clear after a hazard", any("No immediate obstacle" in t for t in said))
    check("buffer learned where the keys were seen", (buf3.where("keys", base + 45) or ("",))[0] == "in the kitchen")
    check("repeated room move became a permanent edge", any(e["from"] == "hallway" for e in s_tmp.edges()))
    graph.invoke({"query": "where are my keys?", "t": base + 46})   # the "question" path through the graph
    check("query path answers through the output agent", "keys" in audio.history[-1][1])

    # ---- Regression test: an object getting more dangerous must NOT be silenced by the repeat-throttle ----
    from .agents.output import OutputAgent

    class _A:  # a tiny fake speaker that just records what it was asked to say
        def __init__(self): self.h = []
        def speak(self, t, p): self.h.append((p, t))

    fa = _A(); oa = OutputAgent(fa)
    for i, (d, p) in enumerate([(2.9, 3), (2.2, 3), (1.5, 2)]):
        oa.run({"t": 100.0 + i, "hazard": {"label": "chair", "zone": "ahead", "distance_m": d, "ttc_s": None, "priority": p}})
    # Expected: caution spoken once, repeat suppressed, then the warning (escalation) spoken.
    check("escalation caution -> warning is spoken", [p for p, _ in fa.h] == [3, 2])

    # ---- Safety test: if the vision model keeps failing, the user must be told (never silent) ----
    class Broken:
        def detect(self, state):
            raise RuntimeError("boom")

    g4, _r4, a4, _s4, _b4 = build(settings, Broken(), True, LocalJsonStore(Path(tempfile.mkdtemp())))
    for step in range(3):
        g4.invoke({"t": base + step})
    check("vision outage is announced", any("unavailable" in t for _, t in a4.history))

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


# ============================================================================
# main(): defines the command-line options and dispatches to run / doctor / selftest.
# ============================================================================
def main() -> None:
    p = argparse.ArgumentParser(prog="gazeassist_agents")
    sub = p.add_subparsers(dest="cmd", required=True)

    # `run` and its options
    r = sub.add_parser("run", help="run the multi-agent pipeline")
    r.add_argument("--source", choices=["mock", "camera", "video"], default="mock",
                   help="mock = scripted walk (no keys); camera = live webcam/glasses feed; video = a video file")
    r.add_argument("--camera", type=int, default=0, help="camera index when --source camera")
    r.add_argument("--video", help="path to a video file when --source video")
    r.add_argument("--hz", type=float, default=1.0, help="frames sent to the agents per second (default 1)")
    r.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until Ctrl+C)")
    r.add_argument("--destination", help="room to guide towards, e.g. kitchen (uses permanent memory)")
    r.add_argument("--console-audio", action="store_true", help="print only; do not speak")
    r.add_argument("--fast", action="store_true", help="mock source only: simulate time without waiting")
    r.add_argument("--interactive", action="store_true", help="type questions/notes while running")
    r.set_defaults(fn=cmd_run)

    # the two helper commands
    sub.add_parser("doctor", help="check keys, packages and audio player").set_defaults(fn=cmd_doctor)
    sub.add_parser("selftest", help="offline checks, no keys needed").set_defaults(fn=cmd_selftest)

    a = p.parse_args()
    if a.cmd == "run" and a.source == "video" and not a.video:
        p.error("--video PATH is required with --source video")
    sys.exit(a.fn(a))   # run the chosen command and use its return value as the exit code
