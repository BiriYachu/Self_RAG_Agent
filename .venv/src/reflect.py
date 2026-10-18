
import json

from src.llm import ask_text


def _parse_json(prompt: str) -> dict:
    """Ask Gemini for JSON and parse its response."""
    response = ask_text(
        prompt
        + "\nReturn only a valid JSON object. Do not use Markdown."
    )

    response = response.strip()

    if response.startswith("```"):
        response = response.split("\n", 1)[1]
        response = response.rsplit("```", 1)[0].strip()

    result = json.loads(response)

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from Gemini.")

    return result


def classify_query(query: str) -> dict:
    """Decide whether to retrieve, answer directly, or clarify."""
    result = _parse_json(
        f"""
        You are the query router for a Self-RAG assistant.

        User query: {query}

        Choose one action:
        - retrieve: The query is clear and needs knowledge-base evidence.
        - direct: The query is clear and can be answered without
          knowledge-base retrieval, such as a greeting.
        - clarify: Essential information is missing or the user's
          intent is genuinely ambiguous.

        Do not choose clarify merely because a query is short.

        Return JSON with these fields:
        {{
          "action": "retrieve, direct, or clarify",
          "clarification_question": "question or empty string"
        }}
        """
    )

    action = result.get("action", "retrieve").lower().strip()

    if action not in {"retrieve", "direct", "clarify"}:
        action = "retrieve"

    question = str(result.get("clarification_question", "")).strip()

    if action == "clarify" and not question:
        action = "retrieve"

    return {
        "action": action,
        "clarification_question": question,
    }


def grade_passages(query: str, passages: list[dict]) -> list[dict]:
    """Judge whether each retrieved passage is relevant."""
    if not passages:
        return []

    result = _parse_json(
        f"""
        Determine whether each passage contains useful information
        for answering the user's query.

        Query: {query}

        Passages:
        {json.dumps(passages, ensure_ascii=False)}

        Return a JSON object with a "grades" array.
        Each grade must contain:
        - "index": the zero-based passage index
        - "relevant": true or false

        Mark a passage relevant only if it provides useful evidence
        for answering the query.
        """
    )

    grades = result.get("grades", [])
    if not isinstance(grades, list):
        return [{"text": p.get("text", ""), "relevant": False}
                for p in passages]

    grade_by_index = {}

    for grade in grades:
        if isinstance(grade, dict):
            index = grade.get("index")
            if isinstance(index, int) and 0 <= index < len(passages):
                grade_by_index[index] = grade.get("relevant") is True

    return [
        {
            **passage,
            "relevant": grade_by_index.get(index, False),
        }
        for index, passage in enumerate(passages)
    ]


def rewrite_query(query: str) -> str:
    """Rewrite a query once to improve retrieval."""
    result = _parse_json(
        f"""
        Rewrite the user's query to improve knowledge-base retrieval.
        Preserve the user's original intent. Do not answer the query.

        User query: {query}

        Return JSON with one field:
        {{"rewritten_query": "improved query"}}
        """
    )

    rewritten = str(result.get("rewritten_query", "")).strip()

    if not rewritten:
        return query

    return rewritten


def critique_answer(
    query: str,
    answer: str,
    passages: list[dict],
) -> dict:
    """Evaluate whether an answer is supported by evidence."""
    result = _parse_json(
        f"""
        Evaluate the answer using only the supplied evidence.

        User query: {query}
        Answer: {answer}

        Evidence:
        {json.dumps(passages, ensure_ascii=False)}

        Assess:
        - is_sup: "fully", "partially", or "no"
        - is_use: an integer from 1 to 5 indicating usefulness

        Use "fully" only when the answer's factual claims are
        supported by the evidence. Use "partially" when only some
        claims are supported. Use "no" when the answer is unsupported.

        Return only JSON with fields "is_sup" and "is_use".
        """
    )

    support = str(result.get("is_sup", "no")).lower().strip()
    if support not in {"fully", "partially", "no"}:
        support = "no"

    try:
        usefulness = int(result.get("is_use", 1))
    except (TypeError, ValueError):
        usefulness = 1

    usefulness = max(1, min(5, usefulness))

    return {
        "is_sup": support,
        "is_use": usefulness,
    }
