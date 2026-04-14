"""
Vektori — Agent-type extraction customisation examples
=======================================================

Shows how to tailor what Vektori extracts at L0 (facts) and L1 (episodes)
for different agent personas.  The same conversation is run through three
differently-configured Vektori instances so you can see the effect.

Quick-start
-----------
    export OPENAI_API_KEY=...
    python examples/agent_type_extraction.py

Levels of customisation
-----------------------
  Level 1 — agent_type preset (zero effort):
      Vektori(agent_type="presales")

  Level 2 — domain hints on top of a preset:
      Vektori(extraction_config=ExtractionConfig(
          agent_type="presales",
          focus_on=["ICP fit", "executive sponsor"],
          ignore=["pleasantries"],
      ))

  Level 3 — prompt suffix (low effort, precise control):
      Vektori(extraction_config=ExtractionConfig(
          agent_type="sales",
          facts_prompt_suffix="Always extract the exact dollar amount when pricing is mentioned.",
      ))

  Level 4 — full prompt override (escape hatch):
      Vektori(extraction_config=ExtractionConfig(
          custom_facts_prompt=MY_PROMPT_TEMPLATE,
      ))
"""

import asyncio
import json

from vektori import ExtractionConfig, Vektori

# ---------------------------------------------------------------------------
# A realistic pre-sales discovery call snippet
# ---------------------------------------------------------------------------
PRESALES_CONVERSATION = [
    {
        "role": "user",
        "content": (
            "We're a Series B fintech, around 200 engineers. "
            "Right now our AI agents lose context between sessions — support keeps "
            "repeating the same questions to customers. It's killing CSAT scores."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "That's a common friction point at your scale. What does your current "
            "session storage look like — are you on something like Redis, or more ad hoc?"
        ),
    },
    {
        "role": "user",
        "content": (
            "Ad hoc, honestly. Each team rolls their own. Our CTO wants a unified "
            "memory layer by Q3 — we've got maybe a $60–80K budget for tooling this half. "
            "We tried Mem0 briefly but the graph retrieval wasn't granular enough."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Got it. So you need something that preserves the full conversation story, "
            "not just entity triples — that's exactly what Vektori's three-layer graph "
            "is built for. Who else is involved in the decision besides your CTO?"
        ),
    },
    {
        "role": "user",
        "content": (
            "Our VP Eng and the platform team lead, Aisha. She's the one who'd actually "
            "integrate it. They're both pretty hands-on technically."
        ),
    },
]

# ---------------------------------------------------------------------------
# A sales closing call
# ---------------------------------------------------------------------------
SALES_CONVERSATION = [
    {
        "role": "user",
        "content": (
            "We reviewed the proposal. Legal flagged the data residency clause — "
            "they need EU hosting confirmed before we can sign."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Understood. Our EU region is live on AWS eu-west-1 — I'll get that "
            "confirmed in writing by tomorrow. Are there any other open items?"
        ),
    },
    {
        "role": "user",
        "content": (
            "Just that. We're looking at the $48K/year enterprise tier. "
            "If legal clears it this week we can sign by Friday the 18th."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Perfect. I'll loop in our legal team tonight. I'll also prep the "
            "countersigned order form so it's ready to go the moment you get approval."
        ),
    },
]


async def run_demo():
    print("=" * 70)
    print("Vektori ExtractionConfig demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Example 1: pre-sales preset — zero effort
    # ------------------------------------------------------------------
    print("\n[1] agent_type='presales' — built-in preset\n")

    presales_v = Vektori(agent_type="presales")
    captured: list[dict] = []
    await presales_v.add(
        messages=PRESALES_CONVERSATION,
        session_id="demo-presales-001",
        user_id="demo-user",
        # _capture_out is an internal debug hook used in tests
    )
    # Give async extraction a moment to complete for the demo
    await asyncio.sleep(3)
    memory = await presales_v.search("budget and decision makers", user_id="demo-user")
    print("Facts retrieved (presales):")
    for f in memory.get("facts", []):
        print(f"  • {f['text']}")
    await presales_v.close()

    # ------------------------------------------------------------------
    # Example 2: sales preset + domain hints
    # ------------------------------------------------------------------
    print("\n[2] agent_type='sales' + focus_on=['contract value', 'close date']\n")

    sales_v = Vektori(
        extraction_config=ExtractionConfig(
            agent_type="sales",
            focus_on=["contract value", "close date", "legal blockers"],
        )
    )
    await sales_v.add(
        messages=SALES_CONVERSATION,
        session_id="demo-sales-001",
        user_id="demo-sales-user",
    )
    await asyncio.sleep(3)
    memory = await sales_v.search("deal status and blockers", user_id="demo-sales-user")
    print("Facts retrieved (sales):")
    for f in memory.get("facts", []):
        print(f"  • {f['text']}")
    await sales_v.close()

    # ------------------------------------------------------------------
    # Example 3: same sales call, general (no preset) — shows the difference
    # ------------------------------------------------------------------
    print("\n[3] agent_type='general' — no domain bias (baseline)\n")

    general_v = Vektori()  # default, no agent_type
    await general_v.add(
        messages=SALES_CONVERSATION,
        session_id="demo-general-001",
        user_id="demo-general-user",
    )
    await asyncio.sleep(3)
    memory = await general_v.search("deal status and blockers", user_id="demo-general-user")
    print("Facts retrieved (general):")
    for f in memory.get("facts", []):
        print(f"  • {f['text']}")
    await general_v.close()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(run_demo())
