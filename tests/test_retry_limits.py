
from unittest.mock import patch

from src.graph import run_agent


def test_query_rewritten_once():
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
        side_effect=[[], [{"text": passages[0]["text"]}]],
    ) as retrieve_mock, patch(
        "src.graph.grade_passages",
        side_effect=[[], passages],
    ), patch(
        "src.graph.rewrite_query",
        return_value="What is the annual leave entitlement?",
    ) as rewrite_mock, patch(
        "src.graph.generate_answer",
        return_value="Employees receive 20 days of annual leave.",
    ), patch(
        "src.graph.critique_answer",
        return_value={"is_sup": "fully", "is_use": 5},
    ):
        result = run_agent("How much leave?")

    assert result["status"] == "completed"
    assert rewrite_mock.call_count == 1
    assert retrieve_mock.call_count == 2


def test_answer_regenerated_once():
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
        side_effect=[
            "Employees receive 30 days of annual leave.",
            "Employees receive 20 days of annual leave.",
        ],
    ) as generate_mock, patch(
        "src.graph.critique_answer",
        side_effect=[
            {"is_sup": "no", "is_use": 2},
            {"is_sup": "fully", "is_use": 5},
        ],
    ):
        result = run_agent("How many annual leave days?")

    assert result["status"] == "completed"
    assert generate_mock.call_count == 2
    assert "20 days" in result["answer"]
