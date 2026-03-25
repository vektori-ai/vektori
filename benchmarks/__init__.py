"""
LongMemEval Benchmark Module

End-to-end evaluation of Vektori's long-term memory capabilities using LongMemEval.

Quick start:
    from benchmarks.longmemeval.longmemeval_runner import BenchmarkConfig, LongMemEvalBenchmark
    
    config = BenchmarkConfig(dataset_name="longmemeval_s_cleaned")
    benchmark = LongMemEvalBenchmark(config)
    await benchmark.run()

Or use the CLI:
    python -m benchmarks.longmemeval.longmemeval_runner --help
    ./benchmarks/longmemeval/run_longmemeval.sh --help
"""

from benchmarks.longmemeval.longmemeval_runner import (
    BenchmarkConfig,
    LongMemEvalBenchmark,
)

__all__ = [
    "BenchmarkConfig",
    "LongMemEvalBenchmark",
]

__version__ = "0.1.0"
