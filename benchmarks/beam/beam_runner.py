"""
BEAM Benchmark Runner for Vektori
=================================

Evaluates Vektori's persistent agent memory and reasoning on the BEAM dataset.
"""

import ast
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from vektori import Vektori
from vektori.config import VektoriConfig
from vektori.models.factory import create_llm
from vektori.qa import generate_answer
from benchmarks.longmemeval.checkpoint import BenchmarkCheckpoint
from benchmarks.beam.beam_judge import BeamJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BeamConfig:
    dataset_split: str = "500K"  # 100K | 500K | 1M
    embedding_model: str = "cloudflare:@cf/baai/bge-m3"
    extraction_model: str = "gemini:gemini-2.5-flash-lite"
    eval_model: str = "gemini:gemini-2.5-flash-lite"
    
    # Reranker integration
    reranker_model: str = "bge:BAAI/bge-reranker-v2-m3"
    top_k: int = 15
    context_window: int = 5
    
    output_dir: str = "benchmark_results"
    run_name: str | None = None
    max_questions: int | None = None


class BeamBenchmark:
    def __init__(self, config: BeamConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vektori_client = None
        self.dataset = []
        self._checkpoint = None
        self._eval_llm = None
        self._run_name = None
        self._empty_chat_logged = False
        self.judge = BeamJudge(config.eval_model)

    async def setup(self) -> None:
        logger.info("Setting up BEAM benchmark environment…")
        
        self.vektori_client = Vektori(
            config=VektoriConfig(
                embedding_model=self.config.embedding_model,
                extraction_model=self.config.extraction_model,
                default_top_k=self.config.top_k,
                context_window=self.config.context_window,
                async_extraction=False,
                use_reranker=True,
                reranker_model=self.config.reranker_model,
                reranker_top_n=max(20, self.config.top_k),
            )
        )
        await self.vektori_client._ensure_initialized()
        self._eval_llm = create_llm(self.config.eval_model, json_mode=False)
        
        logger.info(f"Downloading BEAM dataset {self.config.dataset_split}...")
        hf_ds = load_dataset("Mohammadta/BEAM", split=self.config.dataset_split)
        
        # Load up to max questions limit if needed
        limit = self.config.max_questions if self.config.max_questions else len(hf_ds)
        self.dataset = [hf_ds[i] for i in range(limit)]
        
        self._run_name = self.config.run_name or f"beam_{self.config.dataset_split}"
        chk_path = self.output_dir / f"{self._run_name}_checkpoint.json"
        self._checkpoint = BenchmarkCheckpoint(chk_path)
        n_done = self._checkpoint.load()
        remaining = len(self.dataset) - n_done
        if n_done:
            logger.info("Resuming — %d done, %d remaining", n_done, remaining)
        logger.info(f"Loaded {len(self.dataset)} BEAM instances.")

    async def run(self):
        try:
            await self.setup()
            await self._run_instances()
            await self._evaluate_and_summarize()
        finally:
            if self.vektori_client is not None:
                await self.vektori_client.close()
            logger.info("Run finished.")

    def _parse_chat_messages(self, raw_chat: Any) -> list[dict[str, str]]:
        if isinstance(raw_chat, str):
            try:
                raw_chat = ast.literal_eval(raw_chat)
            except Exception:
                return []

        messages: list[dict[str, str]] = []

        def _collect(node: Any) -> None:
            if node is None:
                return
            if isinstance(node, dict):
                if "role" in node and "content" in node:
                    messages.append({"role": str(node["role"]), "content": str(node["content"])})
                    return
                if "messages" in node:
                    _collect(node.get("messages"))
                    return
                if "text" in node:
                    role = str(node.get("role", "user"))
                    messages.append({"role": role, "content": str(node.get("text", ""))})
                    return

            if isinstance(node, (list, tuple)):
                if len(node) == 2 and not isinstance(node[0], (list, tuple, dict)):
                    messages.append({"role": "user", "content": str(node[0])})
                    messages.append({"role": "assistant", "content": str(node[1])})
                    return
                for item in node:
                    _collect(item)

        _collect(raw_chat)
        return messages

    def _has_pending_questions(self, sample_id: str, probing_questions: dict[str, Any]) -> bool:
        for ability_type, questions in probing_questions.items():
            if isinstance(questions, dict):
                questions = [questions]
            elif not isinstance(questions, list):
                continue
            for sub_idx, _ in enumerate(questions):
                qid = f"{sample_id}_{ability_type}_{sub_idx}"
                if not self._checkpoint.is_done(qid):
                    return True
        return False

    def _format_retrieved_context(self, search_results: dict[str, Any]) -> str:
        if not search_results:
            return "No relevant context retrieved."

        lines: list[str] = []
        for section in ("facts", "episodes", "syntheses", "sentences"):
            items = search_results.get(section) or []
            if not items:
                continue
            lines.append(f"## {section.title()}")
            for i, item in enumerate(items, 1):
                text = str(item.get("text", item)).strip()
                lines.append(f"{i}. {text}")

        return "\n".join(lines) if lines else "No relevant context retrieved."

    async def _run_instances(self):
        for idx, row in enumerate(self.dataset):
            sample_id = f"beam_sample_{idx}"
            # Parse chat turns
            raw_chat = row.get("chat") or row.get("chat_data") or row.get("chat_turns") or row.get("conversation") or []
            chat_turns = self._parse_chat_messages(raw_chat)
            if not chat_turns and not self._empty_chat_logged:
                self._empty_chat_logged = True
                keys = list(row.keys())
                logger.warning(
                    "No chat turns parsed for sample %s. Keys=%s raw_chat_type=%s",
                    sample_id,
                    keys,
                    type(raw_chat).__name__,
                )
            # Parse probing questions (dict of ability_type -> question data)
            probing_questions_str = row.get("probing_questions", "{}")
            if probing_questions_str.startswith("{"):
                probing_questions = ast.literal_eval(probing_questions_str)
            else:
                probing_questions = {}
            
            if not self._has_pending_questions(sample_id, probing_questions):
                continue

            try:
                # 1. Ingestion Phase
                logger.info(f"Ingesting {len(chat_turns)} turns for {sample_id}...")
                
                if chat_turns:
                    await self.vektori_client.add(
                        messages=chat_turns,
                        session_id=sample_id,
                        user_id=sample_id,
                    )

                # 2. Query Phase for each probing question
                for ability_type, questions in probing_questions.items():
                    if isinstance(questions, dict):
                        questions = [questions]
                    elif not isinstance(questions, list):
                        continue
                        
                    for sub_idx, q_dict in enumerate(questions):
                        qid = f"{sample_id}_{ability_type}_{sub_idx}"
                        if self._checkpoint.is_done(qid):
                            continue

                        question_text = q_dict.get("question", "")
                        expected_ans = q_dict.get("answer", "")
                        
                        q_t0 = time.perf_counter()
                        
                        try:
                            # Retrieval + Generation
                            search_results = await self.vektori_client.search(
                                query=question_text,
                                user_id=sample_id,
                                depth="l1",
                            )
                            context = self._format_retrieved_context(search_results)
                            actual_ans = await generate_answer(
                                question=question_text,
                                context=context,
                                question_date="",
                                question_type=str(ability_type),
                                llm=self._eval_llm,
                                prompt_template=None,
                                max_tokens=2048,
                            )
                            score = await self.judge.evaluate_answer(
                                ability_type,
                                question_text,
                                expected_ans,
                                actual_ans,
                            )
                            
                            result = {
                                "question_id": qid,
                                "ability_type": ability_type,
                                "question": question_text,
                                "expected": expected_ans,
                                "generated": actual_ans,
                                "score": score,
                                "latency_ms": (time.perf_counter() - q_t0) * 1000
                            }
                            self._checkpoint.mark_done(qid, result)
                            self._checkpoint.save()
                            logger.info(f"✅ {qid} ({ability_type}) answered. Score: {score}")

                        except Exception as e:
                            logger.error(f"❌ Question {qid} failed: {e}")
                            # Sleep momentarily in case of API rate limits
                            await asyncio.sleep(5)
                            self._checkpoint.mark_failed(qid, str(e))
                            self._checkpoint.save()
                    
            finally:
                # Clean up isolated state
                await self.vektori_client.delete_user(sample_id)

    async def _evaluate_and_summarize(self):
        completed = self._checkpoint.get_completed()
        results = list(completed.values())
        summary = self.judge.compute_summary(results)
        
        sm_path = self.output_dir / f"{self._run_name}_summary.json"
        with open(sm_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved summary to {sm_path}")
