# LongMemEval Benchmark

Run Vektori on LongMemEval benchmark using **BGE embeddings** and **Google Gemini API**.

## Setup

```bash
pip install -e "."
pip install "FlagEmbedding>=1.2" "google-generativeai>=0.3.0" "httpx>=0.27"
export GOOGLE_API_KEY="your-gemini-api-key"
```

Get your Gemini API key free from: https://ai.google.dev/

## Run

```bash
# Basic (LongMemEval-S, L1 depth, BGE + Gemini)
python -m benchmarks.longmemeval.longmemeval_runner \
  --dataset longmemeval_s_cleaned \
  --depth l1

# All depths (L0, L1, L2)
python -m benchmarks.longmemeval.longmemeval_runner \
  --dataset longmemeval_s_cleaned

# Custom config
python -m benchmarks.longmemeval.longmemeval_runner \
  --dataset longmemeval_s_cleaned \
  --depth l1 \
  --embedding-model "bge:BAAI/bge-m3" \
  --extraction-model "gemini:gemini-2-5-flash-lite" \
  --top-k 10 \
  --run-name "my_run"
```

## Python Usage

```python
from benchmarks.longmemeval.longmemeval_runner import BenchmarkConfig, LongMemEvalBenchmark
import asyncio

async def run():
    config = BenchmarkConfig(
        dataset_name="longmemeval_s_cleaned",
        embedding_model="bge:BAAI/bge-m3",
        extraction_model="gemini:gemini-2-5-flash-lite",
    )
    benchmark = LongMemEvalBenchmark(config)
    await benchmark.run()

asyncio.run(run())
```

## Results

- `benchmark_results/{run_name}_summary.json` - Metrics summary
- `benchmark_results/{run_name}_full_results.json` - Complete results
- `benchmark_results/qa_results.jsonl` - Q&A pairs

## Datasets

- `longmemeval_s_cleaned`: ~115k tokens, 40 sessions (fast, ~8-11 min) — **START HERE**
- `longmemeval_m_cleaned`: ~500k tokens, 200+ sessions (slow, ~16-24 min)
- `longmemeval_oracle`: Oracle retrieval baseline (upper bound)

## Retrieval Depths

- `l0`: Facts only (50-200 tokens) — Fastest, cheapest
- `l1`: Facts + Insights (200-500 tokens) — **Recommended, balanced**
- `l2`: Facts + Insights + Sentences (1000-3000 tokens) — Full context, slowest

## Models & Providers

### Embedding (via BGE, fully local)
```
bge:BAAI/bge-m3              # 1024-dim, multilingual (DEFAULT)
bge:BAAI/bge-large-en-v1.5   # 1024-dim, English
bge:BAAI/bge-base-en-v1.5    # Smaller, English
```

### Extraction (via direct Google Gemini API)

**Use `gemini:` prefix with any Gemini model:**

```
# Google Gemini models
gemini:gemini-2-5-flash-lite   # Fast, cost-effective (DEFAULT)
gemini:gemini-2-5-flash
gemini:gemini-3-pro
gemini:gemini-2-flash
gemini:gemini-pro
```

### Alternative: Use other providers
```
# OpenAI
openai:gpt-4o-mini
openai:gpt-4-turbo

# Anthropic
anthropic:claude-haiku-4-5-20251001
anthropic:claude-opus-4-1-20250805

# Ollama (local)
ollama:llama3
ollama:mistral

# LiteLLM (100+ providers)
litellm:groq/llama3-8b-8192
litellm:together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1
```
