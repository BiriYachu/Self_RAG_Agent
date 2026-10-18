
from unittest.mock import patch

from src.reflect import (
    classify_query,
    grade_passages,
    rewrite_query,
    critique_answer,
)


def test_classify_greeting():
    with patch(
        "src.reflect._parse_json",
        return_value={
            "action": "direct",
            "clarification_question": "",
        },
    ):
        result = classify_query("Hello!")

    assert result["action"] == "direct"


def test_classify_ambiguous_query():
    with patch(
        "src.reflect._parse_json",
        return_value={
            "action": "clarify",
            "clarification_question": "Which type of leave?",
        },
    ):
        result = classify_query("Tell me about leave")

    assert result["action"] == "clarify"
    assert result["clarification_question"]


def test_grade_passages():
    passages = [
        {"text": "Employees receive 20 days of annual leave."},
        {"text": "Remote work requires manager approval."},
    ]

    with patch(
        "src.reflect._parse_json",
        return_value={
            "grades": [
                {"index": 0, "relevant": True},
                {"index": 1, "relevant": False},
            ]
        },
    ):
        result = grade_passages("Annual leave entitlement?", passages)

    assert result[0]["relevant"] is True
    assert result[1]["relevant"] is False


def test_rewrite_query():
    with patch(
        "src.reflect._parse_json",
        return_value={
            "rewritten_query": "What is the annual leave allowance?"
        },
    ):
        result = rewrite_query("How much leave?")

    assert result == "What is the annual leave allowance?"


def test_critique_answer():
    with patch(
        "src.reflect._parse_json",
        return_value={
            "is_sup": "fully",
            "is_use": 5,
        },
    ):
        result = critique_answer(
            "How many leave days?",
            "20 days.",
            [{"text": "Employees receive 20 days of annual leave."}],
        )

    assert result["is_sup"] == "fully"
    assert result["is_use"] == 5
