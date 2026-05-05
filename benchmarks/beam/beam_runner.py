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
        self.judge = BeamJudge(config.eval_model)

    async def setup(self) -> None:
        logger.info("Setting up BEAM benchmark environment…")
        
        self.vektori_client = Vektori(
            config=VektoriConfig(
                embedding_model=self.config.embedding_model,
                extraction_model=self.config.extraction_model,
                default_top_k=self.config.top_k,
                context_window=self.config.context_window
                # In full logic, hook up reranker model explicitly if Supported VektoriConfig has it 
            )
        )
        await self.vektori_client._ensure_initialized()
        
        logger.info(f"Downloading BEAM dataset {self.config.dataset_split}...")
        hf_ds = load_dataset("Mohammadta/BEAM", split=self.config.dataset_split)
        
        # Load up to max questions limit if needed
        limit = self.config.max_questions if self.config.max_questions else len(hf_ds)
        self.dataset = [hf_ds[i] for i in range(limit)]
        
        run_name = self.config.run_name or f"beam_{self.config.dataset_split}"
        chk_path = self.output_dir / f"{run_name}_checkpoint.json"
        self._checkpoint = BenchmarkCheckpoint(chk_path)
        logger.info(f"Loaded {len(self.dataset)} BEAM instances.")

    async def run(self):
        try:
            await self.setup()
            await self._run_instances()
            await self._evaluate_and_summarize()
        finally:
            logger.info("Run finished.")

    async def _run_instances(self):
        for idx, row in enumerate(self.dataset):
            # Parse chat turns
            chat_turns = ast.literal_eval(row.get("chat_turns", "[]"))
            # Parse probing questions (dict of ability_type -> question data)
            probing_questions_str = row.get("probing_questions", "{}")
            if probing_questions_str.startswith("{"):
                probing_questions = ast.literal_eval(probing_questions_str)
            else:
                probing_questions = {}

            sample_id = f"beam_sample_{idx}"
            
            # Check if all abilities for this instance are done
            # For simplicity in this runner, if any question is pending, re-run
            all_done = all(self._checkpoint.is_done(f"{sample_id}_{q_idx}") 
                           for q_idx in range(len(probing_questions)))
            
            if all_done:
                continue

            try:
                # 1. Ingestion Phase
                logger.info(f"Ingesting {len(chat_turns)} turns for {sample_id}...")
                
                # Convert raw chat to session format 
                session_messages = [{"role": t.get("role", "user"), "content": t.get("text", "")} for t in chat_turns]
                
                await self.vektori_client._pipeline.ingest(
                    messages=session_messages,
                    session_id=sample_id,
                    user_id=sample_id
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
                            # Emulate standard pipeline flow, including hypothetical reranking step logic
                            reply = await self.vektori_client.process_message(
                                message=question_text,
                                user_id=sample_id,
                                session_id=f"eval_{qid}"
                            )
                            
                            actual_ans = reply.content
                            score = await self.judge.evaluate_answer(ability_type, question_text, expected_ans, actual_ans)
                            
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
        results = list(self._checkpoint.results.values())
        summary = self.judge.compute_summary(results)
        
        sm_path = self.output_dir / f"{self.config.run_name}_summary.json"
        with open(sm_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved summary to {sm_path}")
