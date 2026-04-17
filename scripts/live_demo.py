# -*- coding: utf-8 -*-
"""
Vektori - Live Demo Script
Shows a real benchmark case end-to-end: retrieved context, model answer, verdict.
Uses existing benchmark results — no live API calls.

Usage:
    python scripts/live_demo.py
    python scripts/live_demo.py --case QA_FAILURE   # show a synthesis failure case
    python scripts/live_demo.py --case CORRECT       # show a correct case
    python scripts/live_demo.py --index 0            # show case by index
"""

import io
import json
import sys
import time
import argparse
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
PURPLE = "\033[95m"
BLUE = "\033[94m"
WHITE = "\033[97m"
GRAY = "\033[90m"

RUN_DIR = Path("benchmark_results/locomo_nontemporal_full_queryrewrite_tokens")
JUDGE_FILE = RUN_DIR / "locomo_nontemporal_full_queryrewrite_tokens_judge_results.json"
QA_FILE = RUN_DIR / "locomo_nontemporal_full_queryrewrite_tokens_qa_results.jsonl"
SUMMARY_FILE = RUN_DIR / "locomo_nontemporal_full_queryrewrite_tokens_summary.json"


def bar(label: str, value: float, width: int = 30, color: str = CYAN) -> str:
    filled = int(value * width)
    empty = width - filled
    return f"{color}{'#' * filled}{GRAY}{'.' * empty}{RESET}  {color}{value * 100:.0f}%{RESET}  {GRAY}{label}{RESET}"


def print_divider(char: str = "-", width: int = 65, color: str = GRAY) -> None:
    print(f"{color}{char * width}{RESET}")


def animate_label(label: str, delay: float = 0.3) -> None:
    print(f"\n{GRAY}>{RESET}  {WHITE}{label}{RESET}", flush=True)
    time.sleep(delay)


def load_data():
    with open(JUDGE_FILE) as f:
        judge_cases = json.load(f)
    qa_map = {}
    with open(QA_FILE) as f:
        for line in f:
            r = json.loads(line)
            qa_map[r["question_id"]] = r
    with open(SUMMARY_FILE) as f:
        summary = json.load(f)
    # Enrich judge cases with model answer
    for c in judge_cases:
        qa = qa_map.get(c["question_id"], {})
        c["model_answer"] = qa.get("hypothesis", "")
    return judge_cases, summary


def pick_case(cases, mode: str, index: int | None) -> dict:
    if index is not None:
        return cases[index]
    if mode == "QA_FAILURE":
        matches = [c for c in cases if c.get("failure_mode") == "QA_FAILURE"]
    elif mode == "CORRECT":
        matches = [c for c in cases if c["verdict"] == "CORRECT"]
    else:
        matches = cases
    return matches[0] if matches else cases[0]


def verdict_color(verdict: str) -> str:
    return {
        "CORRECT": GREEN,
        "PARTIALLY_CORRECT": YELLOW,
        "WRONG": RED,
        "ABSTAINED": GRAY,
    }.get(verdict, WHITE)


def render_context_preview(context: str, max_chars: int = 600) -> str:
    lines = context.strip().split("\n")
    out = []
    total = 0
    for line in lines:
        if total + len(line) > max_chars:
            out.append(f"{GRAY}  … (truncated — {len(context)} chars total){RESET}")
            break
        out.append(f"  {GRAY}{line}{RESET}")
        total += len(line)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Vektori live demo")
    parser.add_argument("--case", default="QA_FAILURE", choices=["QA_FAILURE", "CORRECT", "WRONG", "ALL"])
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()

    cases, summary = load_data()
    case = pick_case(cases, args.case, args.index)
    s = summary["metrics"]
    j = s["judge"]
    lat = s["latency_ms"]

    # ── Header ──────────────────────────────────────────────────────────────
    print()
    print_divider("=", 65, PURPLE)
    print(f"  {PURPLE}{BOLD}VEKTORI{RESET}  {WHITE}Benchmark Demo{RESET}  {GRAY}·  LoCoMo Dataset{RESET}")
    print_divider("=", 65, PURPLE)

    # ── Dataset summary ──────────────────────────────────────────────────────
    animate_label("Dataset Overview", 0.2)
    print(f"  {GRAY}Questions  {RESET}{WHITE}{BOLD}{s['total_questions']:,}{RESET}")
    print(f"  {GRAY}Model      {RESET}{WHITE}Gemini 2.5 Flash Lite{RESET}  {GRAY}(production-realistic){RESET}")
    print(f"  {GRAY}Embedding  {RESET}{WHITE}@cf/baai/bge-m3{RESET}  {GRAY}(Cloudflare Workers AI){RESET}")
    print(f"  {GRAY}Depth      {RESET}{WHITE}L1{RESET}  {GRAY}(facts + episodes + sentences){RESET}")

    # ── The key insight ──────────────────────────────────────────────────────
    animate_label("Benchmark Insight (from 100 judged cases)", 0.2)
    print(f"\n  {bar('Context hit rate — retrieval works', j['context_has_answer_rate'], color=CYAN)}")
    print(f"  {bar('QA failure — model had context, failed', j['qa_failure'] / j['n_judged'], color=RED)}")
    print(f"  {bar('Retrieval failure — wrong context', j['retrieval_failure'] / j['n_judged'], color=YELLOW)}")
    print()
    print(f"  {PURPLE}The bottleneck is synthesis, not retrieval.{RESET}")

    # ── Latency ──────────────────────────────────────────────────────────────
    animate_label("Latency Profile", 0.2)
    print(f"  {GRAY}Retrieval   {RESET}{CYAN}{lat['retrieval_ms']['avg']:.0f}ms{RESET} avg  {GRAY}p95 {lat['retrieval_ms']['p95']:.0f}ms{RESET}")
    print(f"  {GRAY}QA gen      {RESET}{YELLOW}{lat['qa_ms']['avg']:.0f}ms{RESET} avg  {GRAY}p95 {lat['qa_ms']['p95']:.0f}ms{RESET}")
    print(f"  {GRAY}End-to-end  {RESET}{WHITE}{lat['total_question_ms']['avg']:.0f}ms{RESET} avg")

    # ── Live case ────────────────────────────────────────────────────────────
    print()
    print_divider("-", 65, GRAY)
    label = "Live Case  —  QA Failure (synthesis gap)" if case.get("failure_mode") == "QA_FAILURE" else "Live Case"
    print(f"  {CYAN}{BOLD}{label}{RESET}")
    print_divider("-", 65, GRAY)

    animate_label("Query", 0.15)
    print(f"  {WHITE}{BOLD}{case['question']}{RESET}")
    print(f"  {GRAY}User: {case['speaker_a']}  ·  Type {case['question_type']}  ·  ID: {case['question_id']}{RESET}")

    animate_label("Retrieval", 0.4)
    print(f"  {CYAN}Facts      {RESET}{WHITE}{case['facts_retrieved']}{RESET}{GRAY} retrieved{RESET}")
    print(f"  {CYAN}Episodes   {RESET}{WHITE}{case['episodes_retrieved']}{RESET}{GRAY} retrieved{RESET}")
    print(f"  {CYAN}Sentences  {RESET}{WHITE}{case['sentences_retrieved']}{RESET}{GRAY} retrieved{RESET}")
    print(f"  {CYAN}Latency    {RESET}{WHITE}{case['retrieval_ms']:.0f}ms{RESET}")

    ctx_has = case.get("context_has_answer", False)
    ctx_label = f"{GREEN}YES — answer is in context{RESET}" if ctx_has else f"{RED}NO — retrieval failed{RESET}"
    print(f"\n  Context has answer:  {ctx_label}")

    animate_label("Retrieved Context (preview)", 0.3)
    print(render_context_preview(case["retrieved_context"]))

    animate_label("QA Generation", 0.4)
    model_ans = case.get("model_answer") or "(abstained / no answer)"
    vc = verdict_color(case["verdict"])
    print(f"  {GRAY}Model answer  {RESET}{vc}{BOLD}{model_ans}{RESET}")
    print(f"  {GRAY}Expected      {RESET}{GREEN}{BOLD}{case['expected_answer']}{RESET}")
    print(f"  {GRAY}Verdict       {RESET}{vc}{BOLD}{case['verdict']}{RESET}")
    print(f"  {GRAY}F1 score      {RESET}{WHITE}{case['f1']:.3f}{RESET}  {GRAY}Exact match: {'yes' if case['exact_match'] else 'no'}{RESET}")
    print(f"  {GRAY}QA latency    {RESET}{WHITE}{case['qa_ms']:.0f}ms{RESET}")

    if case.get("judge_explanation"):
        print(f"\n  {GRAY}Judge: {case['judge_explanation']}{RESET}")

    # ── Failure diagnosis ────────────────────────────────────────────────────
    if case.get("failure_mode") == "QA_FAILURE":
        print()
        print_divider("-", 65, RED)
        print(f"  {RED}{BOLD}Synthesis Failure Confirmed{RESET}")
        print(f"  {GRAY}Retrieval worked. Context contained the answer. Model failed to extract it.{RESET}")
        print(f"  {GRAY}-> This is what GEPA is designed to fix.{RESET}")
        print_divider("-", 65, RED)

    print()
    print_divider("=", 65, PURPLE)
    print(f"  {GRAY}Next: GEPA prompt optimization -> RL-tuned synthesis per vertical{RESET}")
    print_divider("=", 65, PURPLE)
    print()


if __name__ == "__main__":
    main()
