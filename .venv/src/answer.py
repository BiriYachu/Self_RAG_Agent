
from src.llm import ask_text


def generate_answer(query: str, passages: list[dict]) -> str:
    """Generate an answer grounded in retrieved evidence."""

    evidence = "\n\n".join(
        f"Passage {index + 1}: {passage['text']}"
        for index, passage in enumerate(passages)
    )

    prompt = f"""
You are a helpful assistant for Dabba Labs.

Answer the user's question using the supplied evidence.

Rules:
- Use the evidence as your source of truth.
- Do not invent policies or facts.
- If the evidence does not contain the answer, say so clearly.
- Be concise and answer the question directly.

User question:
{query}

Evidence:
{evidence if evidence else "No relevant evidence was retrieved."}

Answer:
"""

    return ask_text(prompt)


def generate_direct_answer(query: str) -> str:
    """Answer a general query without knowledge-base retrieval."""

    prompt = f"""
You are a helpful conversational assistant.

Respond naturally to the user's message.
Do not invent Dabba Labs policies or claim to have checked
the company's handbook.

User message:
{query}

Response:
"""

    return ask_text(prompt)
