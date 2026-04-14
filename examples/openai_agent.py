"""Minimal OpenAI-backed native harness example using `VektoriAgent`."""

import asyncio
from vektori import AgentConfig, Vektori, VektoriAgent
from vektori.models.factory import create_chat_model


async def chat_with_memory(user_id: str):
    memory = Vektori(
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    )
    agent = VektoriAgent(
        memory=memory,
        model=create_chat_model("openai:gpt-4o-mini"),
        user_id=user_id,
        agent_id="openai-agent-demo",
        session_id=f"session-{user_id}-001",
        config=AgentConfig(background_add=True),
    )

    print("Chat with memory (type 'quit' to exit)\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "quit":
                break
            result = await agent.chat(user_input)
            print(f"Assistant: {result.content}\n")
    finally:
        await agent.close()
        await memory.close()


if __name__ == "__main__":
    asyncio.run(chat_with_memory("demo-user"))
