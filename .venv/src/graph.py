
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.config import MAX_REWRITES, MAX_REGENERATES
from src.kb import retrieve
from src.reflect import (
    classify_query,
    grade_passages,
    rewrite_query,
    critique_answer,
)
from src.answer import generate_answer, generate_direct_answer


class AgentState(TypedDict, total=False):
    original_query: str
    query: str
    action: str
    clarification_question: str
    clarification_used: bool
    passages: list[dict]
    answer: str
    rewrite_count: int
    regenerate_count: int
    is_sup: str
    is_use: int
    status: str



def classify_node(state: AgentState) -> dict:
    result = classify_query(state["query"])

    action = result["action"]
    question = result["clarification_question"]

    # Do not ask for clarification more than once.
    if action == "clarify" and state.get("clarification_used", False):
        action = "retrieve"
        question = ""

    return {
        "action": action,
        "clarification_question": question,
        "status": (
            "clarification_needed"
            if action == "clarify"
            else "running"
        ),
    }



def direct_node(state: AgentState) -> dict:
    answer = generate_direct_answer(state["query"])
    return {"answer": answer, "status": "completed"}


def retrieve_node(state: AgentState) -> dict:
    passages = retrieve(state["query"])
    graded = grade_passages(state["query"], passages)
    relevant = [p for p in graded if p.get("relevant")]

    return {"passages": relevant}


def route_after_classify(state: AgentState) -> str:
    action = state.get("action", "retrieve")

    if action == "direct":
        return "direct"

    if action == "clarify":
        return "clarify"

    return "retrieve"


def clarify_node(state: AgentState) -> dict:
    return {
        "status": "clarification_needed",
        "answer": state.get(
            "clarification_question",
            "Could you clarify what you mean?",
        ),
    }


def route_after_retrieve(state: AgentState) -> str:
    if state.get("passages"):
        return "generate"

    if state.get("rewrite_count", 0) < MAX_REWRITES:
        return "rewrite"

    return "not_found"


def rewrite_node(state: AgentState) -> dict:
    rewritten = rewrite_query(state["query"])

    return {
        "query": rewritten,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "passages": [],
    }


def not_found_node(state: AgentState) -> dict:
    return {
        "answer": (
            "I couldn't find sufficiently relevant information "
            "in the knowledge base to answer confidently."
        ),
        "status": "completed",
    }


def generate_node(state: AgentState) -> dict:
    answer = generate_answer(
        state["query"],
        state.get("passages", []),
    )

    critique = critique_answer(
        state["query"],
        answer,
        state.get("passages", []),
    )

    return {
        "answer": answer,
        "is_sup": critique["is_sup"],
        "is_use": critique["is_use"],
    }


def route_after_critique(state: AgentState) -> str:
    supported = state.get("is_sup") == "fully"
    useful = state.get("is_use", 1) >= 3

    if supported and useful:
        return "finish"

    if state.get("regenerate_count", 0) < MAX_REGENERATES:
        return "regenerate"

    return "finish"


def regenerate_node(state: AgentState) -> dict:
    passages = state.get("passages", [])
    answer = generate_answer(state["query"], passages)

    critique = critique_answer(state["query"], answer, passages)

    return {
        "answer": answer,
        "is_sup": critique["is_sup"],
        "is_use": critique["is_use"],
        "regenerate_count": state.get("regenerate_count", 0) + 1,
    }


def finish_node(state: AgentState) -> dict:
    return {"status": "completed"}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("direct", direct_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("not_found", not_found_node)
    builder.add_node("generate", generate_node)
    builder.add_node("regenerate", regenerate_node)
    builder.add_node("finish", finish_node)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "direct": "direct",
            "clarify": "clarify",
            "retrieve": "retrieve",
        },
    )

    builder.add_edge("direct", END)
    builder.add_edge("clarify", END)

    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "not_found": "not_found",
        },
    )

    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("not_found", END)

    builder.add_conditional_edges(
        "generate",
        route_after_critique,
        {
            "finish": "finish",
            "regenerate": "regenerate",
        },
    )

    builder.add_edge("regenerate", "finish")
    builder.add_edge("finish", END)

    return builder.compile()


agent = build_graph()



def run_agent(
    query: str,
    clarification_used: bool = False,
) -> dict:
    """Run the agent for one user query."""

    initial_state: AgentState = {
        "original_query": query,
        "query": query,
        "rewrite_count": 0,
        "regenerate_count": 0,
        "clarification_used": clarification_used,
        "passages": [],
        "status": "running",
    }

    return agent.invoke(initial_state)

