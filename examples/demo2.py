"""
pip install vektori
export OPENAI_API_KEY=sk-...
python examples/sid_demo.py
"""

import asyncio
from vektori import Vektori

USER = "priya-123"


async def main():
    v = Vektori(
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    )

    # ── Session 1: onboarding call ────────────────────────────────────────────
    print("Session 1  (onboarding call, 2 weeks ago)")
    await v.add(
        messages=[
            {"role": "user",      "content": "Please don't call me, I only respond on WhatsApp."},
            {"role": "assistant", "content": "Noted — WhatsApp only for you."},
            {"role": "user",      "content": "Also I'm usually free after 7 PM."},
        ],
        session_id="call-001",
        user_id=USER,
    )
    print("  stored.\n")

    # ── Session 2: follow-up call ─────────────────────────────────────────────
    print("Session 2  (follow-up, yesterday)")
    await v.add(
        messages=[
            {"role": "user",      "content": "I still haven't heard back. I sent a WhatsApp 3 days ago."},
            {"role": "assistant", "content": "Apologies for the delay — I'll follow up now."},
        ],
        session_id="call-002",
        user_id=USER,
    )
    print("  stored.\n")

    await asyncio.sleep(4)  # let async extraction finish

    # ── Query ─────────────────────────────────────────────────────────────────
    print("Search:  'how should we contact this user?'\n")
    results = await v.search(
        query="how should we contact this user?",
        user_id=USER,
        depth="l1",
    )

    print("Facts:")
    for f in results.get("facts", []):
        print(f"  [{f['score']:.2f}]  {f['text']}")

    print("\nEpisodes  (patterns across sessions):")
    for e in results.get("episodes", []):
        print(f"  →  {e['text']}")

    await v.close()


asyncio.run(main())
