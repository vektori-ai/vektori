"""
Realistic seed + latency benchmark for the vektori REST API.

Seeds 20 sessions (~200 messages) then runs p50/p95/p99 latency tests
across l0/l1/l2 depths, both from localhost and reports the numbers.

Usage:
    API_URL=http://13.232.28.196:8000 \
    ADMIN_KEY=my-admin-secret \
    .venv/bin/python tests/seed_and_bench.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time

import httpx

BASE_URL = os.getenv("API_URL", "http://localhost:8000")
ADMIN_KEY = os.getenv("ADMIN_KEY", "my-admin-secret")

# ── Realistic conversation seed data ─────────────────────────────────────────

SESSIONS = [
    {
        "session_id": "s-001",
        "messages": [
            {"role": "user", "content": "I'm a software engineer at a fintech startup in Bangalore."},
            {"role": "assistant", "content": "Got it, I'll remember that."},
            {"role": "user", "content": "I only communicate via WhatsApp, never email."},
            {"role": "assistant", "content": "WhatsApp only, noted."},
        ],
    },
    {
        "session_id": "s-002",
        "messages": [
            {"role": "user", "content": "I'm vegetarian and severely allergic to peanuts."},
            {"role": "assistant", "content": "Vegetarian, peanut allergy — I'll keep that in mind."},
            {"role": "user", "content": "My dog is a golden retriever named Bruno."},
            {"role": "assistant", "content": "Bruno the golden retriever, noted!"},
        ],
    },
    {
        "session_id": "s-003",
        "messages": [
            {"role": "user", "content": "I wake up at 5:30am every day and go for a run before work."},
            {"role": "assistant", "content": "Early riser, morning runner — got it."},
            {"role": "user", "content": "I prefer dark roast coffee, no sugar."},
            {"role": "assistant", "content": "Dark roast, no sugar."},
        ],
    },
    {
        "session_id": "s-004",
        "messages": [
            {"role": "user", "content": "I'm currently learning Rust on weekends. Been at it for 3 months."},
            {"role": "assistant", "content": "Learning Rust, 3 months in."},
            {"role": "user", "content": "My main languages at work are Python and TypeScript."},
            {"role": "assistant", "content": "Python and TypeScript professionally, Rust as a hobby."},
        ],
    },
    {
        "session_id": "s-005",
        "messages": [
            {"role": "user", "content": "I live in Koramangala, Bangalore. Been here for 4 years."},
            {"role": "assistant", "content": "Koramangala, Bangalore for 4 years."},
            {"role": "user", "content": "I commute by metro, takes about 25 minutes each way."},
            {"role": "assistant", "content": "25 min metro commute."},
        ],
    },
    {
        "session_id": "s-006",
        "messages": [
            {"role": "user", "content": "My team uses Slack for work, but I turn off notifications after 7pm."},
            {"role": "assistant", "content": "Slack for work, no notifications after 7pm."},
            {"role": "user", "content": "I block my calendar from 2-4pm for deep work every day."},
            {"role": "assistant", "content": "2-4pm is your deep work block daily."},
        ],
    },
    {
        "session_id": "s-007",
        "messages": [
            {"role": "user", "content": "I'm working on a side project — a SaaS tool for indie developers."},
            {"role": "assistant", "content": "Building a SaaS for indie devs on the side."},
            {"role": "user", "content": "The tech stack is Next.js, Supabase, and Stripe."},
            {"role": "assistant", "content": "Next.js, Supabase, Stripe stack."},
        ],
    },
    {
        "session_id": "s-008",
        "messages": [
            {"role": "user", "content": "I read for 30 minutes every night before sleeping. Currently reading The Pragmatic Programmer."},
            {"role": "assistant", "content": "Nightly reading, currently The Pragmatic Programmer."},
            {"role": "user", "content": "I track my reading on Goodreads. Goal is 24 books this year."},
            {"role": "assistant", "content": "24 book goal, tracked on Goodreads."},
        ],
    },
    {
        "session_id": "s-009",
        "messages": [
            {"role": "user", "content": "I have a standing desk at home and work remotely 3 days a week."},
            {"role": "assistant", "content": "Hybrid work — 3 days remote with standing desk."},
            {"role": "user", "content": "My monitor setup is two 27-inch 4K displays."},
            {"role": "assistant", "content": "Dual 27-inch 4K setup."},
        ],
    },
    {
        "session_id": "s-010",
        "messages": [
            {"role": "user", "content": "I meditate for 10 minutes every morning using Headspace."},
            {"role": "assistant", "content": "Morning meditation with Headspace."},
            {"role": "user", "content": "I've been meditating consistently for about 8 months now."},
            {"role": "assistant", "content": "8 months of consistent meditation."},
        ],
    },
    {
        "session_id": "s-011",
        "messages": [
            {"role": "user", "content": "My salary gets credited on the 1st of every month. I auto-invest 30% immediately."},
            {"role": "assistant", "content": "30% auto-investment on the 1st each month."},
            {"role": "user", "content": "I use Zerodha for stocks and mutual funds."},
            {"role": "assistant", "content": "Zerodha for investments."},
        ],
    },
    {
        "session_id": "s-012",
        "messages": [
            {"role": "user", "content": "I'm introverted and recharge by spending time alone. Large social events drain me."},
            {"role": "assistant", "content": "Introverted — prefers smaller gatherings."},
            {"role": "user", "content": "I have a small group of close friends, maybe 5 people I see regularly."},
            {"role": "assistant", "content": "Small tight-knit friend group of about 5."},
        ],
    },
    {
        "session_id": "s-013",
        "messages": [
            {"role": "user", "content": "I'm planning to move to Berlin in about 18 months for work."},
            {"role": "assistant", "content": "Berlin move planned in ~18 months."},
            {"role": "user", "content": "I've started learning German — currently at A2 level."},
            {"role": "assistant", "content": "German at A2, preparing for Berlin move."},
        ],
    },
    {
        "session_id": "s-014",
        "messages": [
            {"role": "user", "content": "I use Notion for everything — notes, tasks, project management."},
            {"role": "assistant", "content": "Notion as the single tool for notes, tasks, PM."},
            {"role": "user", "content": "I do a weekly review every Sunday evening. Takes about 45 minutes."},
            {"role": "assistant", "content": "Sunday weekly review, 45 min."},
        ],
    },
    {
        "session_id": "s-015",
        "messages": [
            {"role": "user", "content": "I eat out maybe once a week, usually South Indian food on Sundays."},
            {"role": "assistant", "content": "Sunday South Indian meals."},
            {"role": "user", "content": "I cook my own meals for the rest of the week. Meal prep on Sundays."},
            {"role": "assistant", "content": "Sunday meal prep for the week."},
        ],
    },
    {
        "session_id": "s-016",
        "messages": [
            {"role": "user", "content": "My biggest professional goal is to become an engineering manager within 2 years."},
            {"role": "assistant", "content": "EM role targeted within 2 years."},
            {"role": "user", "content": "I'm currently reading about system design and leadership to prepare."},
            {"role": "assistant", "content": "Studying system design and leadership for the EM path."},
        ],
    },
    {
        "session_id": "s-017",
        "messages": [
            {"role": "user", "content": "I have a younger sister who is studying medicine in Pune."},
            {"role": "assistant", "content": "Younger sister studying medicine in Pune."},
            {"role": "user", "content": "We talk every weekend, usually Sunday mornings."},
            {"role": "assistant", "content": "Sunday morning calls with sister."},
        ],
    },
    {
        "session_id": "s-018",
        "messages": [
            {"role": "user", "content": "I play badminton twice a week with colleagues after work."},
            {"role": "assistant", "content": "Badminton twice a week post-work."},
            {"role": "user", "content": "I also do a 5km run on Saturdays, trying to get under 25 minutes."},
            {"role": "assistant", "content": "Saturday 5km, targeting sub-25 min."},
        ],
    },
    {
        "session_id": "s-019",
        "messages": [
            {"role": "user", "content": "I back up all my work to GitHub and also to an external SSD every Friday."},
            {"role": "assistant", "content": "Friday backups to GitHub and external SSD."},
            {"role": "user", "content": "My most important projects are all in private GitHub repos."},
            {"role": "assistant", "content": "Key work in private GitHub repos."},
        ],
    },
    {
        "session_id": "s-020",
        "messages": [
            {"role": "user", "content": "I take a 2-week vacation every year, usually in October to avoid monsoons."},
            {"role": "assistant", "content": "Annual October vacation."},
            {"role": "user", "content": "I prefer solo travel or with one close friend. Hate group tours."},
            {"role": "assistant", "content": "Solo or small travel, avoids group tours."},
        ],
    },
]

SEARCH_QUERIES = [
    ("how should I contact this person?", "WhatsApp"),
    ("what does this user eat?", "vegetarian"),
    ("what programming languages does this person use?", "Python"),
    ("where does this user live?", "Bangalore"),
    ("what are this person's fitness habits?", "run"),
    ("what is this person's career goal?", "manager"),
    ("does this user have any family?", "sister"),
    ("what tools does this person use for productivity?", "Notion"),
    ("what is this person learning?", "Rust"),
    ("what are this person's financial habits?", "invest"),
]


async def create_key(client: httpx.AsyncClient) -> str:
    r = await client.post(
        f"{BASE_URL}/v1/admin/keys",
        json={"name": "bench"},
        headers={"Authorization": f"Bearer {ADMIN_KEY}"},
    )
    r.raise_for_status()
    return r.json()["key"]


async def seed(client: httpx.AsyncClient, key: str, user_id: str) -> None:
    print(f"Seeding {len(SESSIONS)} sessions (2s gap to respect Gemini rate limit)...")
    for i, s in enumerate(SESSIONS):
        r = await client.post(
            f"{BASE_URL}/v1/add",
            json={"messages": s["messages"], "session_id": s["session_id"], "user_id": user_id},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        r.raise_for_status()
        print(f"  {i+1}/{len(SESSIONS)} seeded", end="\r")
        await asyncio.sleep(2)  # ~6 req/min well under free tier limit of 10/min
    print(f"\nSeeded. Waiting 20s for final extraction...")
    await asyncio.sleep(15)


async def bench(client: httpx.AsyncClient, key: str, user_id: str, n: int = 20) -> None:
    print(f"\n{'Depth':<6}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'min':>8}  {'max':>8}")
    print(f"{'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for depth in ("l0", "l1", "l2"):
        times: list[float] = []
        for i in range(n):
            query = SEARCH_QUERIES[i % len(SEARCH_QUERIES)][0]
            t0 = time.perf_counter()
            r = await client.post(
                f"{BASE_URL}/v1/search",
                json={"query": query, "user_id": user_id, "depth": depth, "top_k": 5},
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            elapsed = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            times.append(elapsed)

        s = sorted(times)
        p50 = statistics.median(s)
        p95 = s[int(len(s) * 0.95)]
        p99 = s[int(len(s) * 0.99)]
        print(f"{depth:<6}  {p50:>7.1f}ms  {p95:>7.1f}ms  {p99:>7.1f}ms  {min(s):>7.1f}ms  {max(s):>7.1f}ms")


async def recall(client: httpx.AsyncClient, key: str, user_id: str) -> None:
    print(f"\n{'Query':<45}  {'Expected':<12}  {'Hit?':<5}  {'Top fact'}")
    print(f"{'─'*45}  {'─'*12}  {'─'*5}  {'─'*30}")

    hits = 0
    for query, expected_kw in SEARCH_QUERIES:
        r = await client.post(
            f"{BASE_URL}/v1/search",
            json={"query": query, "user_id": user_id, "depth": "l1", "top_k": 5},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        data = r.json()
        all_text = " ".join(f.get("text", "") for f in data.get("facts", []) + data.get("sentences", []))
        hit = expected_kw.lower() in all_text.lower()
        hits += int(hit)
        top = data["facts"][0]["text"][:40] if data.get("facts") else "—"
        q_short = query[:43] + ".." if len(query) > 45 else query
        print(f"{q_short:<45}  {expected_kw:<12}  {'✓' if hit else '✗':<5}  {top}")

    print(f"\nRecall: {hits}/{len(SEARCH_QUERIES)} = {hits/len(SEARCH_QUERIES)*100:.0f}%")


async def main() -> None:
    print(f"Target: {BASE_URL}\n")

    async with httpx.AsyncClient(timeout=60) as client:
        # Health check
        r = await client.get(f"{BASE_URL}/health")
        r.raise_for_status()
        print(f"API up: {r.json()}")

        key = await create_key(client)
        print(f"Key: {key[:20]}...")
        user_id = "bench-user-01"

        await seed(client, key, user_id)
        print("\n── Latency (ms) ──────────────────────────────")
        await bench(client, key, user_id)
        print("\n── Recall ────────────────────────────────────")
        await recall(client, key, user_id)


if __name__ == "__main__":
    asyncio.run(main())
