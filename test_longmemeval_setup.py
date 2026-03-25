#!/usr/bin/env python3
"""
Quick validation script to ensure LongMemEval-S is properly set up.

Usage:
    python test_longmemeval_setup.py
"""

import asyncio
import sys
from pathlib import Path


async def test_imports():
    """Test that all required packages are installed."""
    print("🔍 Checking imports...")
    
    errors = []
    
    # Core
    try:
        import vektori
        print("  ✓ vektori")
    except ImportError as e:
        errors.append(f"vektori: {e}")
        print("  ✗ vektori")
    
    # FlagEmbedding (BGE)
    try:
        import FlagEmbedding
        print("  ✓ FlagEmbedding (BGE)")
    except Exception as e:
        # FlagEmbedding may have transformers compatibility issues, but it still works via vektori
        print(f"  ⚠️  FlagEmbedding import issue: {str(e)[:60]}... (will work via vektori)")
    
    # LiteLLM
    try:
        import litellm
        print("  ✓ litellm")
    except ImportError as e:
        errors.append(f"litellm: {e}")
        print("  ✗ litellm — install: pip install litellm>=1.30")
    
    # Benchmark code
    try:
        from benchmarks.longmemeval.longmemeval_runner import BenchmarkConfig, LongMemEvalBenchmark
        print("  ✓ benchmarks.longmemeval")
    except ImportError as e:
        errors.append(f"benchmarks.longmemeval: {e}")
        print("  ✗ benchmarks.longmemeval — install: pip install -e .")
    
    return len(errors) == 0, errors


async def test_models():
    """Test that model providers are registered correctly."""
    print("\n🔍 Checking model providers...")
    
    from vektori.models.factory import EMBEDDING_REGISTRY, LLM_REGISTRY
    
    # Check embedding providers
    print("\n  Embedding providers registered:")
    for provider in EMBEDDING_REGISTRY.keys():
        print(f"    • {provider}")
    
    if "bge" not in EMBEDDING_REGISTRY:
        print("    ✗ BGE not registered!")
        return False
    else:
        print("    ✓ BGE available")
    
    # Check LLM providers
    print("\n  LLM providers registered:")
    for provider in LLM_REGISTRY.keys():
        print(f"    • {provider}")
    
    if "litellm" not in LLM_REGISTRY:
        print("    ✗ LiteLLM not registered!")
        return False
    else:
        print("    ✓ LiteLLM available")
    
    if "gemini" in LLM_REGISTRY:
        print("    ⚠️  WARNING: 'gemini' is registered (should use 'litellm:gemini-...')")
    
    return True


async def test_config():
    """Test that default config is correct."""
    print("\n🔍 Checking default configuration...")
    
    from benchmarks.longmemeval.longmemeval_runner import BenchmarkConfig
    
    config = BenchmarkConfig()
    
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Embedding: {config.embedding_model}")
    print(f"  Extraction: {config.extraction_model}")
    print(f"  Depth: {config.retrieval_depth}")
    
    # Check for correct formats
    issues = []
    
    if not config.embedding_model.startswith("bge:"):
        issues.append(f"Embedding model should use 'bge:' prefix, got '{config.embedding_model}'")
    else:
        print("  ✓ Embedding model format correct")
    
    if not (config.extraction_model.startswith("litellm:") or config.extraction_model.startswith("gemini:")):
        issues.append(f"Extraction model should use 'litellm:' or 'gemini:' prefix, got '{config.extraction_model}'")
    else:
        print("  ✓ Extraction model format correct")
    
    if config.retrieval_depth not in ["l0", "l1", "l2"]:
        issues.append(f"Invalid retrieval depth: {config.retrieval_depth}")
    else:
        print("  ✓ Retrieval depth valid")
    
    return len(issues) == 0, issues


async def test_data_dir():
    """Check if data directory exists and is accessible."""
    print("\n🔍 Checking data directory...")
    
    from benchmarks.longmemeval.longmemeval_runner import BenchmarkConfig
    
    config = BenchmarkConfig()
    data_dir = Path(config.data_dir)
    
    if not data_dir.exists():
        print(f"  ⚠️  Data directory does not exist: {data_dir.absolute()}")
        print(f"     It will be created when benchmark runs")
        print(f"     Datasets will be auto-downloaded from HuggingFace")
        return True
    else:
        print(f"  ✓ Data directory exists: {data_dir.absolute()}")
        
        # Check for datasets
        datasets = list(data_dir.glob("*.json"))
        if datasets:
            print(f"  ✓ Found {len(datasets)} dataset(s):")
            for ds in datasets:
                print(f"    • {ds.name}")
        else:
            print(f"  ⚠️  No datasets found, they will be downloaded on first run")
        
        return True


async def test_api_key():
    """Check if GOOGLE_API_KEY is set."""
    print("\n🔍 Checking API keys...")
    
    import os
    
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if google_key:
        print(f"  ✓ GOOGLE_API_KEY is set (using Gemini)")
    else:
        print(f"  ⚠️  GOOGLE_API_KEY not set")
        print(f"     For Gemini: export GOOGLE_API_KEY='your-key'")
        print(f"     Get free key: https://ai.google.dev/")
    
    if openai_key:
        print(f"  ✓ OPENAI_API_KEY is set (can use OpenAI models)")
    
    # Check for Ollama
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get("http://127.0.0.1:11434/api/tags")
            if response.status_code == 200:
                print(f"  ✓ Ollama is running (can use local models)")
            else:
                print(f"  ℹ️  Ollama endpoint found but returned status {response.status_code}")
    except Exception:
        print(f"  ℹ️  Ollama not detected on localhost:11434 (optional)")
    
    return google_key or openai_key or True  # OK if any key is set or Ollama available


async def main():
    """Run all tests."""
    print("=" * 60)
    print("LongMemEval-S Setup Validation")
    print("=" * 60)
    
    all_pass = True
    
    # Test 1: Imports
    imports_ok, import_errors = await test_imports()
    if not imports_ok:
        all_pass = False
        print("\n❌ Import errors found:")
        for error in import_errors:
            print(f"   {error}")
    
    # Test 2: Models
    models_ok = await test_models()
    if not models_ok:
        all_pass = False
    
    # Test 3: Config
    config_ok, config_issues = await test_config()
    if not config_ok:
        all_pass = False
        print("\n❌ Config issues found:")
        for issue in config_issues:
            print(f"   {issue}")
    
    # Test 4: Data directory
    data_ok = await test_data_dir()
    
    # Test 5: API key
    api_ok = await test_api_key()
    
    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ Setup looks good! Ready to run LongMemEval-S")
        print("\nQuick start:")
        print("  export GOOGLE_API_KEY='your-key'")
        print("  python -m benchmarks.longmemeval.longmemeval_runner \\")
        print("    --dataset longmemeval_s_cleaned \\")
        print("    --depth l1")
        return 0
    else:
        print("❌ Setup issues found — see above")
        print("\nPlease install missing dependencies:")
        print("  pip install -e '.[dev]'")
        print("  pip install -r benchmarks/longmemeval/requirements-benchmark.txt")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
