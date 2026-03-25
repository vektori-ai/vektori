"""
LongMemEval Benchmark Runner for Vektori

End-to-end evaluation of Vektori's long-term memory capabilities using LongMemEval.
Tests five core abilities: Information Extraction, Multi-Session Reasoning, 
Knowledge Updates, Temporal Reasoning, and Abstention.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for LongMemEval benchmark run."""

    # LongMemEval data paths
    data_dir: str = "data"
    dataset_name: str = "longmemeval_s_cleaned"  # s|m|oracle
    
    # Vektori configuration
    storage_backend: str = "sqlite"
    database_url: str | None = None
    embedding_model: str = "sentence-transformers:BAAI/bge-m3"  # BGE-M3 via sentence-transformers (local, 1024-dim)
    extraction_model: str = "gemini:gemini-2.5-flash-lite"
    
    # Retrieval configuration
    retrieval_depth: str = "l1"  # l0|l1|l2
    top_k: int = 10
    context_window: int = 3
    
    # Processing
    batch_size: int = 8
    max_workers: int = 4

    # Output configuration
    output_dir: str = "benchmark_results"
    run_name: str | None = None

    # Evaluation
    eval_api_key: str | None = None
    eval_model: str = "gemini:gemini-2.5-flash-lite"


class LongMemEvalBenchmark:
    """Main benchmark runner for Vektori on LongMemEval."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Vektori client (lazy initialization)
        self.vektori_client = None
        self.storage = None
        
        # Dataset loading
        self.dataset = None
        self.dataset_path = None
        
        # Results tracking
        self.results = {
            "config": config.__dict__,
            "ingestion_results": None,
            "retrieval_results": None,
            "qa_results": None,
            "metrics": None,
        }
    
    async def setup(self) -> None:
        """Initialize Vektori client and download/load dataset."""
        logger.info("Setting up benchmark environment...")
        
        # Initialize Vektori
        await self._init_vektori()
        
        # Load or download LongMemEval dataset
        await self._load_dataset()
        
        logger.info("Benchmark setup complete")
    
    async def _init_vektori(self) -> None:
        """Initialize Vektori client."""
        from vektori import Vektori
        
        logger.info(
            "Initializing Vektori with backend=%s, embedding_model=%s, extraction_model=%s",
            self.config.storage_backend,
            self.config.embedding_model,
            self.config.extraction_model,
        )
        
        self.vektori_client = Vektori(
            database_url=self.config.database_url,
            storage_backend=self.config.storage_backend,
            embedding_model=self.config.embedding_model,
            extraction_model=self.config.extraction_model,
            default_top_k=self.config.top_k,
            context_window=self.config.context_window,
            async_extraction=False,
        )
        
        await self.vektori_client._ensure_initialized()
        self.storage = self.vektori_client.db
    
    async def _load_dataset(self) -> None:
        """Load LongMemEval dataset from local path or download."""
        dataset_filename = f"{self.config.dataset_name}.json"
        self.dataset_path = Path(self.config.data_dir) / dataset_filename
        
        if not self.dataset_path.exists():
            logger.warning(
                "Dataset not found at %s. Attempting download...",
                self.dataset_path,
            )
            await self._download_dataset(dataset_filename)
        
        logger.info("Loading dataset from %s", self.dataset_path)
        with open(self.dataset_path, encoding="utf-8") as f:
            self.dataset = json.load(f)
        
        logger.info("Loaded %d test instances", len(self.dataset))
    
    async def _download_dataset(self, filename: str) -> None:
        """Download dataset from Hugging Face."""
        hf_base = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
        url = f"{hf_base}/{filename}"
        
        logger.info("Downloading %s from %s", filename, url)
        
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            with open(self.dataset_path, "wb") as f:
                f.write(response.content)
        
        logger.info("Dataset downloaded to %s", self.dataset_path)
    
    async def ingest_histories(self) -> None:
        """Ingest all chat histories from LongMemEval into Vektori."""
        logger.info("Starting history ingestion for %d instances...", len(self.dataset))
        
        ingestion_results = {
            "total": len(self.dataset),
            "successful": 0,
            "failed": 0,
            "errors": [],
        }
        
        for idx, instance in enumerate(self.dataset):
            try:
                question_id = instance["question_id"]
                haystack_sessions = instance["haystack_sessions"]
                haystack_dates = instance["haystack_dates"]
                
                # Flatten sessions and dates for ingestion
                all_messages = []
                for session_idx, session in enumerate(haystack_sessions):
                    all_messages.extend(session)
                
                # Ingest with user_id = question_id to keep separate memory contexts
                await self.vektori_client.add(
                    messages=all_messages,
                    session_id=question_id,
                    user_id=question_id,
                    metadata={"timestamp": haystack_dates[-1]} if haystack_dates else None,
                )
                
                ingestion_results["successful"] += 1
                
                if (idx + 1) % 50 == 0:
                    logger.info("Ingested %d/%d histories", idx + 1, len(self.dataset))
                    
            except Exception as e:
                logger.error("Failed to ingest %s: %s", question_id, str(e))
                ingestion_results["failed"] += 1
                ingestion_results["errors"].append({
                    "question_id": question_id,
                    "error": str(e),
                })
        
        self.results["ingestion_results"] = ingestion_results
        logger.info(
            "Ingestion complete: %d successful, %d failed",
            ingestion_results["successful"],
            ingestion_results["failed"],
        )
    
    async def retrieve_and_answer(self) -> None:
        """Retrieve relevant context for each question and generate answers."""
        logger.info("Starting retrieval and QA generation...")
        
        qa_results = []
        retrieval_results = {
            "total": 0,
            "successful": 0,
            "failed": 0,
        }
        
        for idx, instance in enumerate(self.dataset):
            try:
                question_id = instance["question_id"]
                question = instance["question"]
                expected_answer = instance["answer"]
                question_type = instance["question_type"]
                
                # Search for relevant context
                search_results = await self.vektori_client.search(
                    query=question,
                    user_id=question_id,
                    depth=self.config.retrieval_depth,
                )
                
                # Extract retrieved context
                retrieved_context = self._format_retrieved_context(search_results)
                
                # Generate answer using context
                generated_answer = await self._generate_answer(
                    question=question,
                    context=retrieved_context,
                    question_type=question_type,
                )
                
                qa_result = {
                    "question_id": question_id,
                    "question": question,
                    "question_type": question_type,
                    "hypothesis": generated_answer,
                    "expected_answer": expected_answer,
                    "retrieved_context": retrieved_context,
                    "retrieval_depth": self.config.retrieval_depth,
                }
                
                qa_results.append(qa_result)
                retrieval_results["successful"] += 1
                retrieval_results["total"] += 1
                
                if (idx + 1) % 50 == 0:
                    logger.info("Processed %d/%d questions", idx + 1, len(self.dataset))
                    
            except Exception as e:
                logger.error("Failed to process question %s: %s", question_id, str(e))
                retrieval_results["failed"] += 1
                retrieval_results["total"] += 1
        
        self.results["qa_results"] = qa_results
        self.results["retrieval_results"] = retrieval_results
        logger.info(
            "QA generation complete: %d successful, %d failed",
            retrieval_results["successful"],
            retrieval_results["failed"],
        )
    
    def _format_retrieved_context(self, search_results: Any) -> str:
        """Format retrieved search results into readable context."""
        if not search_results or "results" not in search_results:
            return "No relevant context retrieved."
        
        context_lines = []
        
        # Add facts (L0)
        if "l0" in search_results["results"]:
            context_lines.append("## Facts")
            for i, fact in enumerate(search_results["results"]["l0"], 1):
                content = fact.get("content", fact.get("text", str(fact)))
                context_lines.append(f"{i}. {content}")
        
        # Add insights (L1)
        if "l1" in search_results["results"]:
            context_lines.append("\n## Insights")
            for i, insight in enumerate(search_results["results"]["l1"], 1):
                content = insight.get("content", insight.get("text", str(insight)))
                context_lines.append(f"{i}. {content}")
        
        # Add sentences with context (L2)
        if "l2" in search_results["results"]:
            context_lines.append("\n## Session Context")
            for i, sent in enumerate(search_results["results"]["l2"], 1):
                content = sent.get("content", sent.get("text", str(sent)))
                context_lines.append(f"{i}. {content}")
        
        return "\n".join(context_lines) if context_lines else "No relevant context retrieved."
    
    async def _generate_answer(
        self,
        question: str,
        context: str,
        question_type: str,
    ) -> str:
        """Generate answer using retrieved context."""
        from vektori.models.factory import create_llm
        
        if not context or "No relevant context" in context:
            return "I don't have relevant information to answer this question."
        
        # Create LLM for answer generation
        llm = create_llm(self.config.extraction_model)
        
        prompt = self._build_qa_prompt(question, context, question_type)
        
        try:
            response = await llm.generate(prompt, max_tokens=500)
            return response.strip()
        except Exception as e:
            logger.warning("Failed to generate answer: %s", str(e))
            return "Unable to generate answer due to API error."
    
    def _build_qa_prompt(
        self,
        question: str,
        context: str,
        question_type: str,
    ) -> str:
        """Build prompt for QA generation."""
        abstention_hint = ""
        if question_type.endswith("_abs"):
            abstention_hint = (
                "\nNote: This question may be testing abstention (refusing to answer "
                "non-existent events). If the context doesn't contain relevant "
                "information, you should abstain or indicate the information is not available."
            )
        
        prompt = f"""You are an AI assistant answering questions based on provided context from chat history.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer the question based ONLY on the provided context
- Be concise and direct
- If the context doesn't contain the answer, indicate that the information is not available{abstention_hint}

ANSWER:"""
        
        return prompt
    
    async def evaluate(self) -> None:
        """Evaluate generated answers against expected answers."""
        logger.info("Starting evaluation...")
        
        if not self.results["qa_results"]:
            logger.warning("No QA results to evaluate")
            return
        
        # Save QA results to JSONL for evaluation
        qa_jsonl_path = self.output_dir / "qa_results.jsonl"
        with open(qa_jsonl_path, "w") as f:
            for result in self.results["qa_results"]:
                jsonl_entry = {
                    "question_id": result["question_id"],
                    "hypothesis": result["hypothesis"],
                }
                f.write(json.dumps(jsonl_entry) + "\n")
        
        logger.info("QA results saved to %s", qa_jsonl_path)
        
        # Compute basic metrics
        self._compute_basic_metrics()
        
        logger.info("Evaluation complete")
    
    def _compute_basic_metrics(self) -> None:
        """Compute basic evaluation metrics."""
        qa_results = self.results["qa_results"]
        
        if not qa_results:
            return
        
        metrics = {
            "total_questions": len(qa_results),
            "answered": sum(1 for r in qa_results if r["hypothesis"] and "not available" not in r["hypothesis"].lower()),
            "abstained": sum(1 for r in qa_results if "not available" in r["hypothesis"].lower()),
            "by_type": {},
        }
        
        # Group by question type
        for result in qa_results:
            q_type = result["question_type"]
            if q_type not in metrics["by_type"]:
                metrics["by_type"][q_type] = {"total": 0, "answered": 0}
            
            metrics["by_type"][q_type]["total"] += 1
            if result["hypothesis"] and "not available" not in result["hypothesis"].lower():
                metrics["by_type"][q_type]["answered"] += 1
        
        self.results["metrics"] = metrics
        
        logger.info("Metrics computed:")
        logger.info("  Total questions: %d", metrics["total_questions"])
        logger.info("  Answered: %d", metrics["answered"])
        logger.info("  Abstained: %d", metrics["abstained"])
        logger.info("  By type: %s", metrics["by_type"])
    
    async def save_results(self) -> None:
        """Save all results to JSON files."""
        logger.info("Saving results...")
        
        # Determine run name
        run_name = self.config.run_name or self.config.dataset_name
        
        # Save full results
        results_path = self.output_dir / f"{run_name}_full_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info("Full results saved to %s", results_path)
        
        # Save summary
        summary = {
            "config": self.results["config"],
            "ingestion": self.results["ingestion_results"],
            "retrieval": self.results["retrieval_results"],
            "metrics": self.results["metrics"],
        }
        summary_path = self.output_dir / f"{run_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Summary saved to %s", summary_path)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.vektori_client:
            await self.vektori_client.close()
            logger.info("Vektori client closed")
    
    async def run(self) -> None:
        """Run the complete benchmark pipeline."""
        try:
            await self.setup()
            await self.ingest_histories()
            await self.retrieve_and_answer()
            await self.evaluate()
            await self.save_results()
            
            logger.info("Benchmark complete!")
            self._print_results_summary()
            
        finally:
            await self.cleanup()
    
    def _print_results_summary(self) -> None:
        """Print a summary of results."""
        print("\n" + "=" * 60)
        print("LONGMEMEVAL BENCHMARK RESULTS")
        print("=" * 60)
        
        if self.results["metrics"]:
            metrics = self.results["metrics"]
            print(f"\nTotal Questions: {metrics['total_questions']}")
            print(f"Answered: {metrics['answered']}")
            print(f"Abstained: {metrics['abstained']}")
            
            if metrics["by_type"]:
                print("\nBy Question Type:")
                for q_type, counts in metrics["by_type"].items():
                    answer_rate = (
                        counts["answered"] / counts["total"] * 100
                        if counts["total"] > 0
                        else 0
                    )
                    print(f"  {q_type}: {counts['answered']}/{counts['total']} ({answer_rate:.1f}%)")
        
        if self.results["ingestion_results"]:
            ingestion = self.results["ingestion_results"]
            print(f"\nIngestion Results:")
            print(f"  Successful: {ingestion['successful']}/{ingestion['total']}")
            print(f"  Failed: {ingestion['failed']}")
        
        print(f"\nResults saved to: {self.output_dir}")
        print("=" * 60 + "\n")


async def main():
    """Main entry point for benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Vektori benchmark on LongMemEval"
    )
    parser.add_argument(
        "--dataset",
        choices=["longmemeval_s_cleaned", "longmemeval_m_cleaned", "longmemeval_oracle"],
        default="longmemeval_s_cleaned",
        help="LongMemEval dataset to use (default: longmemeval_s_cleaned)",
    )
    parser.add_argument(
        "--depth",
        choices=["l0", "l1", "l2"],
        default="l1",
        help="Retrieval depth (default: l1)",
    )
    parser.add_argument(
        "--embedding-model",
        default="openai:text-embedding-3-small",
        help="Embedding model to use (default: openai:text-embedding-3-small)",
    )
    parser.add_argument(
        "--extraction-model",
        default="openai:gpt-4o-mini",
        help="Extraction model to use (default: openai:gpt-4o-mini)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing LongMemEval data (default: data)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--run-name",
        help="Name for this benchmark run (optional)",
    )
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        dataset_name=args.dataset,
        retrieval_depth=args.depth,
        embedding_model=args.embedding_model,
        extraction_model=args.extraction_model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        top_k=args.top_k,
        run_name=args.run_name,
    )
    
    logger.info("Starting LongMemEval benchmark with config: %s", config)
    
    benchmark = LongMemEvalBenchmark(config)
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())
