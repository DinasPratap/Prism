"""The memory router: decides WHICH memory a fact belongs to and WHICH memory answers a question.

Tiered so it stays fast and cheap:
  tier 1  regex rules                      (instant, free)
  tier 2  Gemini structured classification (only when the rules cannot decide)
  promotion  things seen in the same place on 3 separate occasions, and room-to-room moves seen twice,
             are promoted from the RAM buffer into permanent memory (Firebase).

Vision observations are processed on a background thread so memory work never delays an alert.
"""
from __future__ import annotations

import queue
import re
import sys
import threading
import time

from ..schemas import MemoryDecision, QueryIntent
from .buffer import RecentBuffer, ago, norm
from .permanent import PermanentStore, plan_route

ITEM_RE = re.compile(
    r"\b(?:put|kept|keep|left|leave|placed|place|dropped)\s+(?:my\s+|the\s+)?"
    r"(?P<item>[a-z][\w' -]{1,30}?)\s+"
    r"(?P<prep>on|in|at|under|near|beside|inside|next to|behind)\s+"
    r"(?P<art>the\s+|my\s+|a\s+)?(?P<place>[a-z][\w' -]{1,40}?)\s*[.!]?$",
    re.I,
)
ROUTE_RE = re.compile(
    r"\bfrom\s+(?:the\s+|my\s+)?(?P<a>[a-z][\w' -]{1,30}?)\s+(?:to|->)\s+(?:the\s+|my\s+)?"
    r"(?P<b>[a-z][\w' -]{1,30}?)(?=\s*[:,;.]|\s+(?:go|walk|take|turn|it|is|then)\b|$)",
    re.I,
)
ROUTE_WORDS = re.compile(r"\b(steps?|turn|straight|left|right|door|hall|corridor|stairs)\b", re.I)
Q_ITEM = re.compile(r"where\s+(?:are|is|did i (?:put|leave|keep))\s+(?:my|the)\s+(?P<item>[\w' -]+?)\s*\??$", re.I)
Q_ROUTE = re.compile(
    r"(?:how (?:do i|to|can i) (?:get|go|walk)|take me|directions?|route|navigate)\s*(?:to|me to)?\s*"
    r"(?:the\s+)?(?P<dest>[\w' -]+?)\s*\??$",
    re.I,
)
Q_RECENT = re.compile(r"where (?:have i been|was i)|recently|last room", re.I)


class MemoryRouter:
    def __init__(self, buffer: RecentBuffer, store: PermanentStore, llm=None, promote_after: int = 3):
        self.buf, self.store, self.llm = buffer, store, llm
        self.promote_after = promote_after
        self._sightings: dict[tuple[str, str], list[float]] = {}   # (item, place) -> [count, last_counted_t]
        self._moves: dict[tuple[str, str], int] = {}
        self._pending_room: str | None = None
        self._pending_n = 0
        self._edge_cache: tuple[float, list[dict]] = (0.0, [])
        self._q: queue.Queue = queue.Queue(maxsize=8)
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------------ vision-driven writes
    def observe(self, room: str | None, items: list[str], t: float) -> None:
        """Non-blocking: drops the observation if the worker is busy (memory work is best-effort)."""
        if not room and not items:
            return
        try:
            self._q.put_nowait((room, items, t))
        except queue.Full:
            pass

    def flush(self) -> None:
        self._q.join()

    def _worker(self) -> None:
        while True:
            room, items, t = self._q.get()
            try:
                self._observe(room, items, t)
            except Exception as exc:
                print(f"[memory] write failed: {type(exc).__name__}", file=sys.stderr)
            finally:
                self._q.task_done()

    def _observe(self, room: str | None, items: list[str], t: float) -> None:
        if room:  # debounce: a room must be seen twice in a row before it counts as a visit
            if room == self._pending_room:
                self._pending_n += 1
            else:
                self._pending_room, self._pending_n = room, 1
            if self._pending_n >= 2 and room != self.buf.current_room():
                prev = self.buf.visit(room, t)
                if prev:
                    n = self._moves[(prev, room)] = self._moves.get((prev, room), 0) + 1
                    if n == 2:  # repeated move -> permanent topology
                        self.store.put_edge(prev, room, None)
                        self._edge_cache = (0.0, [])
        here = self.buf.current_room() or room
        for item in items:
            place = f"in the {here}" if here else "somewhere nearby"
            self.buf.see_item(item, place, t)
            rec = self._sightings.setdefault((norm(item), place), [0, -1e12])
            if t - rec[1] >= 600:  # count distinct occasions, not frames
                rec[0], rec[1] = rec[0] + 1, t
                if rec[0] >= self.promote_after:
                    self.store.put_habit(norm(item), place, int(rec[0]))
        if len(self._sightings) > 200:
            self._sightings.clear()

    # ------------------------------------------------------------------ user notes
    def decide(self, text: str) -> MemoryDecision:
        m = ROUTE_RE.search(text)
        if m and ROUTE_WORDS.search(text):
            steps = text[m.end():].strip(" :,.;") or text
            return MemoryDecision(store="permanent", kind="route", from_room=norm(m["a"]), to_room=norm(m["b"]), steps=steps)
        m = ITEM_RE.search(text)
        if m:
            art = (m["art"] or "the ").lower()
            return MemoryDecision(store="buffer", kind="item_location", item=norm(m["item"]),
                                  place=f"{m['prep'].lower()} {art}{m['place'].strip()}")
        if self.llm is not None:
            try:
                prompt = ("Classify this note from a blind person's assistant. Notes about where a personal item "
                          "was put are store=buffer/kind=item_location (set item, place). Directions between rooms "
                          "are store=permanent/kind=route (set from_room, to_room, steps). Anything else is "
                          f"store=discard.\nNote: {text}")
                return self.llm.with_structured_output(MemoryDecision).invoke(prompt)
            except Exception as exc:
                print(f"[memory] router LLM failed: {type(exc).__name__}", file=sys.stderr)
        return MemoryDecision(store="discard", kind="other")

    def remember(self, text: str, t: float | None = None) -> str:
        t = time.time() if t is None else t
        d = self.decide(text)
        if d.store == "buffer" and d.kind == "item_location" and d.item and d.place:
            self.buf.see_item(d.item, d.place, t)
            return f"Okay, I will remember your {d.item} {'are' if d.item.endswith('s') else 'is'} {d.place}."
        if d.store == "permanent" and d.kind == "route" and d.from_room and d.to_room:
            self.store.put_edge(norm(d.from_room), norm(d.to_room), d.steps)
            self._edge_cache = (0.0, [])
            return f"Saved the route from {d.from_room} to {d.to_room}."
        return "I did not store that."

    # ------------------------------------------------------------------ reads
    def current_room(self) -> str | None:
        return self.buf.current_room()

    def _edges(self) -> list[dict]:
        ts, cached = self._edge_cache
        if time.time() - ts > 60 or not cached:
            cached = self.store.edges()
            self._edge_cache = (time.time(), cached)
        return cached

    def route_to(self, room: str, dest: str) -> str | None:
        path = plan_route(self._edges(), norm(room), norm(dest))
        if path is None:
            return None
        if not path:
            return "You are already there."
        return " Then ".join((e.get("steps") or f"go from {e['from']} to {e['to']}").rstrip(".") + "." for e in path)

    def answer(self, query: str, t: float | None = None) -> str:
        t = time.time() if t is None else t
        q = query.strip()
        if m := Q_ITEM.search(q):
            return self._answer_item(m["item"], t)
        if m := Q_ROUTE.search(q):
            return self._answer_route(m["dest"])
        if Q_RECENT.search(q):
            rooms = self.buf.recent_rooms()
            return ("Recently you were in: " + ", then ".join(rooms) + ".") if rooms else "I have no recent movements."
        intent = self._llm_intent(q)
        if intent.kind == "item" and intent.item:
            return self._answer_item(intent.item, t)
        if intent.kind == "route" and intent.destination:
            return self._answer_route(intent.destination)
        if intent.kind == "recent":
            rooms = self.buf.recent_rooms()
            return ("Recently you were in: " + ", then ".join(rooms) + ".") if rooms else "I have no recent movements."
        return "Sorry, I do not know how to answer that."

    def _llm_intent(self, q: str) -> QueryIntent:
        if self.llm is None:
            return QueryIntent(kind="unknown")
        try:
            return self.llm.with_structured_output(QueryIntent).invoke(
                "Classify this question to a home assistant: item = where an object is, route = directions to "
                f"a room, recent = where the person recently went, else unknown.\nQuestion: {q}")
        except Exception:
            return QueryIntent(kind="unknown")

    def _answer_item(self, item: str, t: float) -> str:
        hit = self.buf.where(item, t)
        if hit:
            return f"I last noted your {norm(item)} {hit[0]}, {ago(hit[1])}."
        habit = self.store.habits().get(norm(item))
        if habit:
            return f"I have not seen your {norm(item)} recently. It is usually {habit['place']}."
        return f"I do not have a memory of your {norm(item)}."

    def _answer_route(self, dest: str) -> str:
        room = self.current_room()
        if not room:
            return "I do not know which room you are in yet."
        route = self.route_to(room, dest)
        return f"To get to the {norm(dest)}: {route}" if route else f"I do not know a route from the {room} to the {norm(dest)} yet."
