"""Minimal native harness demo built on top of Vektori."""

from __future__ import annotations

import asyncio

from vektori import Vektori, VektoriAgent
from vektori.models.factory import create_chat_model


async def main() -> None:
    memory = Vektori(
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    )
    agent = VektoriAgent(
        memory=memory,
        model=create_chat_model("openai:gpt-4o-mini"),
        user_id="demo-user",
        agent_id="demo-agent",
    )

    result = await agent.chat("What do you remember about how I like answers?")
    print(result.content)
    await agent.close()
    await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
