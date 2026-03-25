# LongMemEval Benchmark

Run Vektori on LongMemEval benchmark using **BGE embeddings** and **Google Gemini Flash 2.5 Lite**.

## Setup

```bash
pip install -r benchmarks/longmemeval/requirements-benchmark.txt
export GOOGLE_API_KEY="your-gemini-api-key"
```

Get your Gemini API key from: https://ai.google.dev/

## Run

```bash
cd benchmarks/longmemeval

# Basic
./run_longmemeval.sh --dataset longmemeval_s_cleaned --depth l1

# All depths
./run_longmemeval.sh --dataset longmemeval_s_cleaned --all-depths

# Options
--dataset {longmemeval_s_cleaned, longmemeval_m_cleaned, longmemeval_oracle}
--depth {l0, l1, l2}
--embedding-model "bge:bge-large-en-v1.5" (default - BGE)
--extraction-model "gemini:gemini-2.0-flash-lite" (default - Gemini)
--top-k 10
--run-name "my_run"
```

## Python Usage

```python
from benchmarks.longmemeval_runner import BenchmarkConfig, LongMemEvalBenchmark
import asyncio

async def run():
    config = BenchmarkConfig(dataset_name="longmemeval_s_cleaned")
    benchmark = LongMemEvalBenchmark(config)
    await benchmark.run()

asyncio.run(run())
```

## Results

- `benchmark_results/{run_name}_summary.json` - Metrics
- `benchmark_results/{run_name}_full_results.json` - Complete results
- `benchmark_results/qa_results.jsonl` - Q&A pairs

## Datasets

- `longmemeval_s_cleaned`: ~115k tokens, 40 sessions (fast, ~8-11 min)
- `longmemeval_m_cleaned`: ~500 sessions (slow, ~16-24 min)
- `longmemeval_oracle`: Oracle retrieval (upper bound)

## Retrieval Depths

- `l0`: Facts only (50-200 tokens)
- `l1`: Facts + Insights (200-500 tokens, **recommended**)
- `l2`: Facts + Insights + Sentences (1000-3000 tokens)
