"""
Comprehensive local API test suite.

Usage:
    # Make sure docker-compose is up first:
    # docker-compose up -d --build

    .venv/bin/python tests/test_api.py

Covers:
  1. Auth (wrong key, no key, bad format)
  2. Multi-tenant isolation (Key A can't see Key B's memories)
  3. Latency (p50 / p95 / p99 on /v1/search at l0, l1, l2)
  4. Accuracy (add known facts → search with paraphrases → recall score)
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from typing import Any

import httpx

BASE_URL = os.getenv("API_URL", "http://localhost:8000")
ADMIN_KEY = os.getenv("ADMIN_KEY", "my-admin-secret")

# ── Helpers ───────────────────────────────────────────────────────────────────

def header(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


async def create_key(client: httpx.AsyncClient, name: str) -> str:
    r = await client.post(
        f"{BASE_URL}/v1/admin/keys",
        json={"name": name},
        headers=header(ADMIN_KEY),
    )
    r.raise_for_status()
    return r.json()["key"]


def ok(label: str) -> None:
    print(f"  ✓  {label}")


def fail(label: str, detail: str = "") -> None:
    print(f"  ✗  {label}" + (f" — {detail}" if detail else ""))


# ── Test 1: Auth ──────────────────────────────────────────────────────────────

async def test_auth(client: httpx.AsyncClient) -> None:
    print("\n── Auth ──────────────────────────────────────────────────────")

    # Health is unauthenticated
    r = await client.get(f"{BASE_URL}/health")
    assert r.status_code == 200, r.text
    ok("GET /health → 200 (no auth needed)")

    # No key
    r = await client.post(f"{BASE_URL}/v1/search", json={"query": "x", "user_id": "u1"})
    assert r.status_code == 401, f"expected 401 got {r.status_code}"
    ok("No key → 401")

    # Wrong key
    r = await client.post(
        f"{BASE_URL}/v1/search",
        json={"query": "x", "user_id": "u1"},
        headers=header("vk_sk_notreal"),
    )
    assert r.status_code == 401
    ok("Wrong key → 401")

    # Wrong admin key
    r = await client.post(
        f"{BASE_URL}/v1/admin/keys",
        json={"name": "test"},
        headers=header("wrong-admin"),
    )
    assert r.status_code == 401
    ok("Wrong admin key → 401")

    # Valid admin key creates a key
    key = await create_key(client, "auth-test")
    assert key.startswith("vk_sk_")
    ok(f"Admin creates key → {key[:14]}...")

    # That key works
    r = await client.post(
        f"{BASE_URL}/v1/search",
        json={"query": "hello", "user_id": "u1"},
        headers=header(key),
    )
    assert r.status_code == 200
    ok("New key works → 200")


# ── Test 2: Multi-tenant isolation ────────────────────────────────────────────

async def test_isolation(client: httpx.AsyncClient) -> None:
    print("\n── Multi-tenant isolation ────────────────────────────────────")

    key_a = await create_key(client, "tenant-a")
    key_b = await create_key(client, "tenant-b")
    ok(f"Created key A ({key_a[:14]}...) and key B ({key_b[:14]}...)")

    # Tenant A stores a very specific fact under user "alice"
    secret = "Alice's secret PIN is 9281 and she loves purple elephants"
    r = await client.post(
        f"{BASE_URL}/v1/add",
        json={
            "messages": [
                {"role": "user", "content": secret},
                {"role": "assistant", "content": "Got it, I'll remember that."},
            ],
            "session_id": "isolation-test-001",
            "user_id": "alice",
        },
        headers=header(key_a),
    )
    assert r.status_code == 200
    ok("Tenant A stored secret fact for user 'alice'")

    # Wait for async extraction
    await asyncio.sleep(4)

    # Tenant A can find it
    r = await client.post(
        f"{BASE_URL}/v1/search",
        json={"query": "purple elephants PIN", "user_id": "alice", "depth": "l0"},
        headers=header(key_a),
    )
    assert r.status_code == 200
    facts_a = r.json().get("facts", [])
    found_in_a = any("purple" in f.get("text", "").lower() or "9281" in f.get("text", "") for f in facts_a)

    if found_in_a:
        ok("Tenant A can retrieve their own fact ✓")
    else:
        fail("Tenant A could NOT find their own fact", f"got {len(facts_a)} facts")

    # Tenant B searches same user_id — must get nothing
    r = await client.post(
        f"{BASE_URL}/v1/search",
        json={"query": "purple elephants PIN", "user_id": "alice", "depth": "l0"},
        headers=header(key_b),
    )
    assert r.status_code == 200
    facts_b = r.json().get("facts", [])
    leaked = any("purple" in f.get("text", "").lower() or "9281" in f.get("text", "") for f in facts_b)

    if not leaked:
        ok("Tenant B gets ZERO results for same user_id — isolation holds ✓")
    else:
        fail("ISOLATION BREACH — Tenant B can see Tenant A's data!", str(facts_b))


# ── Test 3: Latency ───────────────────────────────────────────────────────────

async def test_latency(client: httpx.AsyncClient) -> None:
    print("\n── Latency ───────────────────────────────────────────────────")

    key = await create_key(client, "latency-test")

    # Seed some memories first
    for i in range(5):
        await client.post(
            f"{BASE_URL}/v1/add",
            json={
                "messages": [
                    {"role": "user", "content": f"I enjoy hiking and outdoor activities session {i}"},
                    {"role": "assistant", "content": "Great, I'll remember that."},
                ],
                "session_id": f"lat-seed-{i}",
                "user_id": "latency-user",
            },
            headers=header(key),
        )
    ok("Seeded 5 sessions — waiting for extraction...")
    await asyncio.sleep(6)

    async def measure(depth: str, n: int = 20) -> list[float]:
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = await client.post(
                f"{BASE_URL}/v1/search",
                json={"query": "outdoor activities hobbies", "user_id": "latency-user", "depth": depth},
                headers=header(key),
            )
            elapsed = (time.perf_counter() - t0) * 1000
            assert r.status_code == 200
            times.append(elapsed)
        return times

    print(f"\n  {'Depth':<6}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'min':>8}  {'max':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for depth in ("l0", "l1", "l2"):
        times = await measure(depth)
        times_sorted = sorted(times)
        p50 = statistics.median(times_sorted)
        p95 = times_sorted[int(len(times_sorted) * 0.95)]
        p99 = times_sorted[int(len(times_sorted) * 0.99)]
        print(
            f"  {depth:<6}  {p50:>7.0f}ms  {p95:>7.0f}ms  {p99:>7.0f}ms  "
            f"{min(times_sorted):>7.0f}ms  {max(times_sorted):>7.0f}ms"
        )

    # expand=True latency (extra LLM call)
    times = await measure("l1", n=5)
    r_times: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = await client.post(
            f"{BASE_URL}/v1/search",
            json={"query": "outdoor activities", "user_id": "latency-user", "expand": True},
            headers=header(key),
        )
        r_times.append((time.perf_counter() - t0) * 1000)
    print(f"  {'expand':<6}  {statistics.median(r_times):>7.0f}ms  (p50, 5 samples — uses extra LLM call)")


# ── Test 4: Accuracy ─────────────────────────────────────────────────────────

KNOWN_FACTS: list[dict[str, Any]] = [
    {
        "fact": "User only uses WhatsApp, never email",
        "paraphrases": ["communication preference", "how to contact user", "messaging app"],
    },
    {
        "fact": "User is vegetarian and allergic to peanuts",
        "paraphrases": ["dietary restrictions", "food allergy", "what can user eat"],
    },
    {
        "fact": "User works as a software engineer at a fintech startup",
        "paraphrases": ["job", "occupation", "where does user work"],
    },
    {
        "fact": "User lives in San Francisco and commutes by bike",
        "paraphrases": ["location", "where does user live", "how does user get around"],
    },
    {
        "fact": "User's dog is named Max, a golden retriever",
        "paraphrases": ["pet", "dog name", "animal"],
    },
]


async def test_accuracy(client: httpx.AsyncClient) -> None:
    print("\n── Accuracy ──────────────────────────────────────────────────")

    key = await create_key(client, "accuracy-test")

    # Ingest all known facts as a single conversation
    messages = []
    for item in KNOWN_FACTS:
        messages.append({"role": "user", "content": item["fact"]})
        messages.append({"role": "assistant", "content": "Noted, I'll remember that."})

    r = await client.post(
        f"{BASE_URL}/v1/add",
        json={"messages": messages, "session_id": "accuracy-seed-001", "user_id": "acc-user"},
        headers=header(key),
    )
    assert r.status_code == 200
    ok(f"Ingested {len(KNOWN_FACTS)} known facts — waiting for extraction...")
    await asyncio.sleep(8)

    total_queries = 0
    hits = 0

    print(f"\n  {'Fact':<40}  {'Query':<35}  {'Hit?'}")
    print(f"  {'─'*40}  {'─'*35}  {'─'*4}")

    for item in KNOWN_FACTS:
        fact_text = item["fact"]
        keywords = [w.lower() for w in fact_text.split() if len(w) > 4]

        for query in item["paraphrases"]:
            r = await client.post(
                f"{BASE_URL}/v1/search",
                json={"query": query, "user_id": "acc-user", "depth": "l1", "top_k": 5},
                headers=header(key),
            )
            assert r.status_code == 200
            data = r.json()

            all_text = " ".join(
                f.get("text", "").lower()
                for f in data.get("facts", []) + data.get("sentences", [])
            )
            hit = any(kw in all_text for kw in keywords[:3])
            hits += int(hit)
            total_queries += 1

            fact_short = fact_text[:38] + ".." if len(fact_text) > 40 else fact_text
            query_short = query[:33] + ".." if len(query) > 35 else query
            print(f"  {fact_short:<40}  {query_short:<35}  {'✓' if hit else '✗'}")

    recall = hits / total_queries * 100
    print(f"\n  Recall: {hits}/{total_queries} = {recall:.0f}%")
    if recall >= 80:
        ok(f"Accuracy good ({recall:.0f}% recall)")
    elif recall >= 60:
        print(f"  ⚠  Accuracy marginal ({recall:.0f}% recall) — check extraction logs")
    else:
        fail(f"Accuracy poor ({recall:.0f}% recall) — extraction may not be working")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 60)
    print("  Vektori API — Local Test Suite")
    print(f"  Target: {BASE_URL}")
    print("=" * 60)

    # Check server is up
    try:
        async with httpx.AsyncClient(timeout=5) as probe:
            r = await probe.get(f"{BASE_URL}/health")
            r.raise_for_status()
    except Exception as e:
        print(f"\n  ERROR: API not reachable at {BASE_URL}")
        print(f"  {e}")
        print("\n  Run: docker-compose up -d --build")
        return

    ok(f"API reachable at {BASE_URL}")

    async with httpx.AsyncClient(timeout=30) as client:
        await test_auth(client)
        await test_isolation(client)
        await test_latency(client)
        await test_accuracy(client)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
