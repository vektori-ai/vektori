"""
Vektori as memory for a basic OpenAI agent.

Each turn: search memory → inject context → respond → store new messages.
"""

import asyncio
import os
from openai import AsyncOpenAI
from vektori import Vektori

client = AsyncOpenAI()


async def chat_with_memory(user_id: str):
    v = Vektori(
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    )

    session_id = f"session-{user_id}-001"
    conversation_history = []

    print("Chat with memory (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # 1. Retrieve relevant memories
        memory = await v.search(query=user_input, user_id=user_id, depth="l1")
        facts_context = "\n".join(f"- {f['text']}" for f in memory.get("facts", []))
        episodes_context = "\n".join(f"- {ep['text']}" for ep in memory.get("episodes", []))

        # 2. Build system prompt with memory context
        system = "You are a helpful assistant with access to user memory.\n"
        if facts_context:
            system += f"\nKnown facts about this user:\n{facts_context}"
        if episodes_context:
            system += f"\nMemory episodes:\n{episodes_context}"

        # 3. Get response
        conversation_history.append({"role": "user", "content": user_input})
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, *conversation_history],
        )
        assistant_reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        print(f"Assistant: {assistant_reply}\n")

        # 4. Store this exchange in memory
        await v.add(
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_reply},
            ],
            session_id=session_id,
            user_id=user_id,
        )

    await v.close()


if __name__ == "__main__":
    asyncio.run(chat_with_memory("demo-user"))
