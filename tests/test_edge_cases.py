
import json
from unittest.mock import patch

import pytest

from src import kb, reflect
from src.graph import run_agent


def test_empty_query_returns_prompt_without_llm_calls():
    with patch("src.graph.classify_query") as classify_mock:
        for query in ["", "   ", "\n"]:
            result = run_agent(query)
            assert result["status"] == "completed"
            assert "question" in result["answer"].lower()

    classify_mock.assert_not_called()


def test_not_found_after_one_rewrite():
    with patch(
        "src.graph.classify_query",
        return_value={"action": "retrieve", "clarification_question": ""},
    ), patch(
        "src.graph.retrieve", return_value=[{"text": "irrelevant"}]
    ) as retrieve_mock, patch(
        "src.graph.grade_passages",
        return_value=[{"text": "irrelevant", "relevant": False}],
    ), patch(
        "src.graph.rewrite_query", return_value="rewritten"
    ) as rewrite_mock, patch(
        "src.graph.generate_answer"
    ) as generate_mock:
        result = run_agent("What is the CEO's favourite colour?")

    assert result["status"] == "completed"
    assert "couldn't find" in result["answer"]
    assert rewrite_mock.call_count == 1
    assert retrieve_mock.call_count == 2
    generate_mock.assert_not_called()


def test_clarification_not_repeated():
    with patch(
        "src.graph.classify_query",
        return_value={
            "action": "clarify",
            "clarification_question": "Which leave?",
        },
    ), patch("src.graph.retrieve", return_value=[]), patch(
        "src.graph.grade_passages", return_value=[]
    ), patch("src.graph.rewrite_query", return_value="q"):
        result = run_agent("leave", clarification_used=True)

    assert result["status"] == "completed"


def test_unsupported_answer_still_finishes_after_one_regeneration():
    passages = [{"text": "20 days", "relevant": True}]

    with patch(
        "src.graph.classify_query",
        return_value={"action": "retrieve", "clarification_question": ""},
    ), patch("src.graph.retrieve", return_value=passages), patch(
        "src.graph.grade_passages", return_value=passages
    ), patch(
        "src.graph.generate_answer", return_value="bad"
    ) as generate_mock, patch(
        "src.graph.critique_answer",
        return_value={"is_sup": "no", "is_use": 1},
    ):
        result = run_agent("How many days?")

    assert result["status"] == "completed"
    assert generate_mock.call_count == 2


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"action": "direct", "clarification_question": ""}\n```',
        'Sure! {"action": "direct", "clarification_question": ""} Done.',
    ],
)
def test_parse_json_handles_fences_and_prose(raw):
    with patch("src.reflect.ask_text", return_value=raw):
        assert reflect._parse_json("p")["action"] == "direct"


def test_parse_json_rejects_non_json():
    with patch("src.reflect.ask_text", return_value="no json here"):
        with pytest.raises(ValueError):
            reflect._parse_json("p")


def test_reflect_functions_survive_bad_model_output():
    passages = [{"text": "a"}, {"text": "b"}]

    with patch("src.reflect._parse_json", side_effect=ValueError("bad")):
        assert reflect.classify_query("hi")["action"] == "retrieve"
        assert reflect.rewrite_query("hi") == "hi"
        assert all(
            p["relevant"] for p in reflect.grade_passages("q", passages)
        )
        assert reflect.critique_answer("q", "a", passages) == {
            "is_sup": "no",
            "is_use": 1,
        }


def test_classify_clarify_without_question_falls_back_to_retrieve():
    with patch(
        "src.reflect._parse_json",
        return_value={"action": "clarify", "clarification_question": ""},
    ):
        assert reflect.classify_query("x")["action"] == "retrieve"


def test_critique_clamps_values():
    with patch(
        "src.reflect._parse_json",
        return_value={"is_sup": "FULLY", "is_use": 99},
    ):
        assert reflect.critique_answer("q", "a", []) == {
            "is_sup": "fully",
            "is_use": 5,
        }

    with patch(
        "src.reflect._parse_json",
        return_value={"is_sup": "maybe", "is_use": "abc"},
    ):
        assert reflect.critique_answer("q", "a", []) == {
            "is_sup": "no",
            "is_use": 1,
        }


def test_grade_passages_empty_list_makes_no_call():
    with patch("src.reflect._parse_json") as parse_mock:
        assert reflect.grade_passages("q", []) == []
    parse_mock.assert_not_called()


def test_retrieve_empty_query_returns_nothing():
    assert kb.retrieve("") == []
    assert kb.retrieve("   ") == []
    assert kb.retrieve("leave", top_k=0) == []
