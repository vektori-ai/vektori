"""Local Vektori demo using Ollama + Milvus.

Prerequisites:
  1) Ollama running locally with:
       - nomic-embed-text
       - qwen2.5:1.5b (or adjust EXTRACTION_MODEL)
  2) Milvus available at http://localhost:19530

Run:
  python examples/ollama_milvus_session_demo.py
"""

from __future__ import annotations

import asyncio

from vektori import Vektori


EMBEDDING_MODEL = "ollama:nomic-embed-text"
EXTRACTION_MODEL = "ollama:qwen2.5:1.5b"
MILVUS_URL = "http://localhost:19530"


async def main() -> None:
    # nomic-embed-text returns 768-dim vectors; Milvus schema must match.
    client = Vektori(
        storage_backend="milvus",
        database_url=MILVUS_URL,
        embedding_model=EMBEDDING_MODEL,
        extraction_model=EXTRACTION_MODEL,
        embedding_dimension=768,
        async_extraction=False,
    )

    user_id = "demo-user-ollama-milvus"
    session_id = "session-ollama-milvus-001"

    messages = [
        {
            "role": "user",
            "content": (
                "I live in Pune and work remotely as a backend engineer. "
                "I prefer calls between 9am and 11am IST, and avoid meetings after 6pm."
            ),
        },
        {
            "role": "assistant",
            "content": "Understood. I will schedule calls in your preferred morning window.",
        },
        {
            "role": "user",
            "content": "I also like concise weekly updates every Monday morning.",
        },
    ]

    try:
        ingest_result = await client.add(
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            metadata={"source": "local-demo"},
        )
        print("Ingestion:", ingest_result)

        search_result = await client.search(
            query="When should I contact this user and what communication style do they prefer?",
            user_id=user_id,
            depth="l1",
        )

        facts = search_result.get("facts", [])
        episodes = search_result.get("episodes", [])
        sentences = search_result.get("sentences", [])

        print("\nFacts:")
        for fact in facts:
            print(f" - {fact.get('text')}")

        print("\nEpisodes:")
        for episode in episodes:
            print(f" - {episode.get('text')}")

        print("\nSentences:")
        for sentence in sentences[:5]:
            print(f" - {sentence.get('text')}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
