
from pathlib import Path

import chromadb

from src.config import TOP_K


# Store the database inside the project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "chroma"

# Create a persistent ChromaDB client.
client = chromadb.PersistentClient(path=str(DB_PATH))

# Create or open our collection.
collection = client.get_or_create_collection(
    name="dabba_labs_handbook"
)


# Example handbook documents for our demo company.
HANDBOOK = [
    {
        "id": "leave_001",
        "text": (
            "Dabba Labs employees receive 20 days of "
            "annual leave per calendar year."
        ),
        "topic": "annual_leave",
    },
    {
        "id": "leave_002",
        "text": (
            "Employees should submit planned annual leave "
            "requests to their manager for approval."
        ),
        "topic": "leave_approval",
    },
    {
        "id": "sick_001",
        "text": (
            "Employees can take sick leave when they are "
            "unwell. They should notify their manager "
            "as soon as reasonably possible."
        ),
        "topic": "sick_leave",
    },
    {
        "id": "remote_001",
        "text": (
            "Remote work requires manager approval and "
            "must follow the team's work arrangements."
        ),
        "topic": "remote_work",
    },
    {
        "id": "expense_001",
        "text": (
            "Employees must submit business expense "
            "claims with receipts through the finance "
            "process."
        ),
        "topic": "expenses",
    },
]


def seed_knowledge_base() -> None:
    """Insert or update the example handbook documents."""

    collection.upsert(
        ids=[item["id"] for item in HANDBOOK],
        documents=[item["text"] for item in HANDBOOK],
        metadatas=[
            {"topic": item["topic"]}
            for item in HANDBOOK
        ],
    )


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the most relevant handbook passages."""

    # Make sure the example documents are available.
    seed_knowledge_base()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    return [
        {
            "text": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document, metadata, distance in zip(
            documents, metadatas, distances
        )
    ]
