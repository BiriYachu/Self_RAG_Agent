
from src.graph import run_agent


def main():
    print("=" * 50)
    print("       Dabba Labs — Self-RAG Assistant")
    print("=" * 50)
    print("Ask a question, or type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.\n")
            continue

        try:
            result = run_agent(query)

            if result.get("status") == "clarification_needed":
                clarification = result.get(
                    "clarification_question",
                    result.get("answer", "Could you clarify your question?"),
                )

                print(f"\nAssistant: {clarification}")

                try:
                    reply = input("You (clarification): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not reply:
                    print("No clarification provided. Try again.\n")
                    continue

                clarified_query = (
                    f"Original question: {query}\n"
                    f"Clarification: {reply}"
                )

                # Ask for clarification at most once per question.
                result = run_agent(clarified_query, clarification_used=True)

            print(f"\nAssistant: {result.get('answer', 'No answer returned.')}\n")

        except Exception as exc:
            print(f"\nAn error occurred: {exc}\n")


if __name__ == "__main__":
    main()
