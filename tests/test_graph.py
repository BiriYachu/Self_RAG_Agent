
from unittest.mock import patch

from src.graph import run_agent


def test_direct_answer():
    with patch(
        "src.graph.classify_query",
        return_value={
            "action": "direct",
            "clarification_question": "",
        },
    ), patch(
        "src.graph.generate_direct_answer",
        return_value="Hello! How can I help?",
    ):
        result = run_agent("Hello!")

    assert result["status"] == "completed"
    assert result["answer"] == "Hello! How can I help?"


def test_retrieval_answer():
    passages = [
        {
            "text": "Employees receive 20 days of annual leave.",
            "relevant": True,
        }
    ]

    with patch(
        "src.graph.classify_query",
        return_value={
            "action": "retrieve",
            "clarification_question": "",
        },
    ), patch(
        "src.graph.retrieve",
        return_value=passages,
    ), patch(
        "src.graph.grade_passages",
        return_value=passages,
    ), patch(
        "src.graph.generate_answer",
        return_value="Employees receive 20 days of annual leave.",
    ), patch(
        "src.graph.critique_answer",
        return_value={"is_sup": "fully", "is_use": 5},
    ):
        result = run_agent(
            "How many annual leave days do employees receive?"
        )

    assert result["status"] == "completed"
    assert "20 days" in result["answer"]


def test_clarification_requested():
    with patch(
        "src.graph.classify_query",
        return_value={
            "action": "clarify",
            "clarification_question": "Which type of leave?",
        },
    ):
        result = run_agent("Tell me about leave")

    assert result["status"] == "clarification_needed"
    assert result["clarification_question"] == "Which type of leave?"
