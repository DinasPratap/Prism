"""LangGraph wiring.

            START ── query? ──yes──► memory_read ─────────────────────────┐
              │ no                                                         ▼
              ▼                                  ┌─► path   (agent 2) ─┐
           obstacle (agent 1) ───────────────────┼─► risk   (agent 3) ─┼─► output (agent 4) ─► END
                                                 └─► memory (router)  ─┘

path, risk and memory run in parallel. No checkpointer is used on purpose: the graph keeps no history,
so memory use is flat. The only long-lived state is the two memory stores.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.obstacle import ObstacleAgent
from .agents.output import OutputAgent
from .agents.path import PathAgent
from .agents.risk import RiskAgent
from .schemas import AssistState


def build_graph(vision_backend, router, audio):
    obstacle = ObstacleAgent(vision_backend)
    path = PathAgent(router)
    risk = RiskAgent()
    output = OutputAgent(audio)

    def memory_write(state: AssistState) -> dict:
        router.observe(state.get("room"), state.get("items") or [], state.get("t", 0.0))
        return {"mem_queued": True}  # returns instantly; the router works on its own thread

    def memory_read(state: AssistState) -> dict:
        return {"memory_answer": router.answer(state["query"], state.get("t"))}

    g = StateGraph(AssistState)
    g.add_node("obstacle", obstacle.run)
    g.add_node("path", path.run)
    g.add_node("risk", risk.run)
    g.add_node("memory", memory_write)
    g.add_node("memory_read", memory_read)
    g.add_node("output", output.run)

    g.add_conditional_edges(
        START,
        lambda s: "memory_read" if s.get("query") else "obstacle",
        {"memory_read": "memory_read", "obstacle": "obstacle"},
    )
    for node in ("path", "risk", "memory"):
        g.add_edge("obstacle", node)
    g.add_edge(["path", "risk", "memory"], "output")
    g.add_edge("memory_read", "output")
    g.add_edge("output", END)
    return g.compile()
