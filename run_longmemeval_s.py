#!/usr/bin/env python3
"""
LongMemEval-S Benchmark Runner with Fault Tolerance

Pre-configured for:
  - BGE embeddings (local, 1024-dim, no API)
  - Google Gemini 2.5 Flash Lite (direct API, free tier)
  - Automatic retry with exponential backoff
  - SQLite storage (zero setup)

Usage:
  python run_longmemeval_s.py                  # Run with defaults
  python run_longmemeval_s.py --verify_only   # Check setup only
  python run_longmemeval_s.py --batch_size 4  # Adjust batch size

Requirements:
  - GOOGLE_API_KEY environment variable set
  - python -m vektori (pip install -e .)
  - FlagEmbedding>=1.2, google-generativeai>=0.3

Setup:
  1. Get free API key: https://ai.google.dev/
  2. Set key: $env:GOOGLE_API_KEY = 'your-key'
  3. Run: python run_longmemeval_s.py
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path


async def main():
    """Run LongMemEval-S benchmark with Gemini + BGE."""
    
    parser = argparse.ArgumentParser(description="LongMemEval-S Benchmark Runner")
    parser.add_argument("--verify_only", action="store_true", help="Check setup without running")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--workers", type=int, default=4, help="Async workers (default: 4)")
    parser.add_argument("--depth", default="l1", choices=["l0", "l1", "l2"], help="Retrieval depth")
    parser.add_argument("--dataset", default="longmemeval_s_cleaned", help="Dataset name")
    args = parser.parse_args()
    
    # Check API key early
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not set")
        print("\nSet it with:")
        print("  PowerShell: $env:GOOGLE_API_KEY = 'your-key'")
        print("  Bash:       export GOOGLE_API_KEY='your-key'")
        print("\nGet free key at: https://ai.google.dev/")
        sys.exit(1)
    
    # Import after validation
    try:
        from benchmarks.longmemeval.longmemeval_runner import (
            BenchmarkConfig,
            LongMemEvalBenchmark,
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nInstall missing packages:")
        print("  pip install -e '.'")
        print("  pip install FlagEmbedding>=1.2 google-generativeai>=0.3 httpx")
        sys.exit(1)
    
    # Configuration with fault tolerance
    config = BenchmarkConfig(
        # Dataset
        dataset_name=args.dataset,
        data_dir="data",
        
        # Models with fault tolerance built-in
        embedding_model="sentence-transformers:BAAI/bge-m3",  # BGE-M3 via sentence-transformers (local, 1024-dim)
        extraction_model="gemini:gemini-2.5-flash-lite",
        
        # Retrieval
        retrieval_depth=args.depth,  # L0=facts, L1=insights, L2=full
        top_k=10,
        context_window=3,
        
        # Storage
        storage_backend="sqlite",  # Auto-created, no setup needed
        
        # Processing
        batch_size=args.batch_size,
        max_workers=args.workers,
        
        # Output
        output_dir="benchmark_results",
        run_name="longmemeval_s_run",
    )
    
    # Print configuration
    print("=" * 70)
    print("🚀 LongMemEval-S Benchmark (Fault-Tolerant Mode)")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"   Dataset:      {config.dataset_name}")
    print(f"   Embedding:    {config.embedding_model} (local, no API key)")
    print(f"   Extraction:   {config.extraction_model}")
    print(f"   Depth:        {config.retrieval_depth} (facts + insights)")
    print(f"   Batch Size:   {config.batch_size}")
    print(f"   Workers:      {config.max_workers}")
    print(f"   Storage:      {config.storage_backend}")
    print(f"   Output:       {config.output_dir}/")
    print(f"\n⚙️  Fault Tolerance:")
    print(f"   - Auto-retry: 5 attempts per API call")
    print(f"   - Backoff:    1s → 2s → 4s → 8s → 16s (+jitter)")
    print(f"   - Handles:    Rate limits (429), timeouts, service errors")
    print(f"\n⏱️  Expected time: ~8-11 minutes for S dataset\n")
    
    if args.verify_only:
        print("✅ Configuration verified. Run without --verify_only to start benchmark.")
        return
    
    # Run benchmark with error handling
    benchmark = LongMemEvalBenchmark(config)
    try:
        await benchmark.run()
        
        print("\n" + "=" * 70)
        print("✅ Benchmark complete!")
        print(f"📁 Results saved to: {config.output_dir}/")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Benchmark failed: {e}")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("  • Check GOOGLE_API_KEY is set correctly")
        print("  • Verify API key has Gemini API access")
        print("  • Check internet connection")
        print("  • Try lower batch size: --batch_size 2")
        print("  • Check logs for 'Retry' messages (shows retry logic working)")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
