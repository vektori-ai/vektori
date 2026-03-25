#!/bin/bash

###############################################################################
# LongMemEval Benchmark Runner for Vektori
#
# Usage:
#   ./benchmarks/run_longmemeval.sh [OPTIONS]
#
# Examples:
#   ./benchmarks/run_longmemeval.sh --dataset longmemeval_s_cleaned --depth l1
#   ./benchmarks/run_longmemeval.sh --all-depths
#   ./benchmarks/run_longmemeval.sh --setup-only
#
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATASET="longmemeval_s_cleaned"
DEPTH="l1"
EMBEDDING_MODEL="openai:text-embedding-3-small"
EXTRACTION_MODEL="openai:gpt-4o-mini"
OUTPUT_DIR="benchmark_results"
DATA_DIR="data"
TOP_K=10
RUN_NAME=""
SETUP_ONLY=false
ALL_DEPTHS=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to print help
print_help() {
    cat << EOF
${BLUE}LongMemEval Benchmark Runner for Vektori${NC}

Usage: ./benchmarks/run_longmemeval.sh [OPTIONS]

Options:
    --dataset DATASET           LongMemEval dataset to use
                               (default: longmemeval_s_cleaned)
                               Options: longmemeval_s_cleaned, 
                                       longmemeval_m_cleaned,
                                       longmemeval_oracle

    --depth DEPTH              Retrieval depth (default: l1)
                               Options: l0, l1, l2

    --all-depths               Run benchmark for all depths (l0, l1, l2)

    --embedding-model MODEL    Embedding model (default: openai:text-embedding-3-small)

    --extraction-model MODEL   Extraction model (default: openai:gpt-4o-mini)

    --top-k K                  Number of top results (default: 10)

    --run-name NAME            Name for this run (optional)

    --output-dir DIR           Output directory (default: benchmark_results)

    --data-dir DIR             Data directory (default: data)

    --setup-only               Only check setup, don't run benchmark

    --help                     Show this help message

Examples:
    # Quick test with S dataset, L1 retrieval
    ./benchmarks/run_longmemeval.sh \\
        --dataset longmemeval_s_cleaned \\
        --depth l1

    # Test all retrieval depths
    ./benchmarks/run_longmemeval.sh \\
        --dataset longmemeval_s_cleaned \\
        --all-depths

    # Comprehensive evaluation
    ./benchmarks/run_longmemeval.sh \\
        --dataset longmemeval_m_cleaned \\
        --depth l2 \\
        --embedding-model "openai:text-embedding-3-large" \\
        --run-name "vektori_comprehensive"

    # Check setup without running
    ./benchmarks/run_longmemeval.sh --setup-only

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --depth)
            DEPTH="$2"
            shift 2
            ;;
        --all-depths)
            ALL_DEPTHS=true
            shift
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --extraction-model)
            EXTRACTION_MODEL="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

###############################################################################
# Main Script
###############################################################################

print_info "═══════════════════════════════════════════════════════════════"
print_info "LongMemEval Benchmark Runner for Vektori"
print_info "═══════════════════════════════════════════════════════════════"

# Check prerequisites
print_info "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi
print_success "Python found: $(python3 --version)"

# Check API key
if [ -z "$GOOGLE_API_KEY" ]; then
    print_error "GOOGLE_API_KEY environment variable not set"
    print_info "Set it with: export GOOGLE_API_KEY='your-gemini-api-key'"
    print_info "Get API key from: https://ai.google.dev/"
    exit 1
fi
print_success "GOOGLE_API_KEY configured"

# Check dependencies
print_info "Checking Python dependencies..."
python3 << 'PYEOF'
import sys
deps = ["vektori", "httpx", "openai", "pydantic"]
missing = []
for dep in deps:
    try:
        __import__(dep)
    except ImportError:
        missing.append(dep)

if missing:
    print(f"Missing dependencies: {', '.join(missing)}")
    print("Install with: pip install -r benchmarks/requirements-benchmark.txt")
    sys.exit(1)
else:
    print("All dependencies installed")
PYEOF

if [ $? -ne 0 ]; then
    exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"
print_success "Directories ready"

if [ "$SETUP_ONLY" = true ]; then
    print_success "Setup check passed! Ready to run benchmarks."
    exit 0
fi

# Determine run name if not specified
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="vektori_${DATASET}_${DEPTH}"
fi

print_info "═══════════════════════════════════════════════════════════════"
print_info "Configuration"
print_info "═══════════════════════════════════════════════════════════════"
print_info "Dataset: $DATASET"
print_info "Depth: $DEPTH"
print_info "Embedding Model: $EMBEDDING_MODEL"
print_info "Extraction Model: $EXTRACTION_MODEL"
print_info "Top-K: $TOP_K"
print_info "Run Name: $RUN_NAME"
print_info "Output Directory: $OUTPUT_DIR"
print_info "Data Directory: $DATA_DIR"

# Function to run benchmark
run_benchmark() {
    local dataset=$1
    local depth=$2
    local run_name=$3
    
    print_info "─────────────────────────────────────────────────────────────"
    print_info "Running benchmark: ${run_name}"
    print_info "─────────────────────────────────────────────────────────────"
    
    python3 -m benchmarks.longmemeval.longmemeval_runner \
        --dataset "$dataset" \
        --depth "$depth" \
        --embedding-model "$EMBEDDING_MODEL" \
        --extraction-model "$EXTRACTION_MODEL" \
        --top-k "$TOP_K" \
        --run-name "$run_name" \
        --output-dir "$OUTPUT_DIR" \
        --data-dir "$DATA_DIR"
    
    if [ $? -eq 0 ]; then
        print_success "Benchmark completed: ${run_name}"
    else
        print_error "Benchmark failed: ${run_name}"
        return 1
    fi
}

# Run benchmark(s)
print_info "═══════════════════════════════════════════════════════════════"
print_info "Starting Benchmark Execution"
print_info "═══════════════════════════════════════════════════════════════"

if [ "$ALL_DEPTHS" = true ]; then
    # Run all depths
    for depth in l0 l1 l2; do
        run_name="${DATASET}_${depth}"
        run_benchmark "$DATASET" "$depth" "$run_name"
        if [ $? -ne 0 ]; then
            print_error "Benchmark failed at depth: $depth"
            exit 1
        fi
    done
else
    # Run single depth
    run_benchmark "$DATASET" "$DEPTH" "$RUN_NAME"
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

# Print completion summary
print_info "═══════════════════════════════════════════════════════════════"
print_success "All benchmarks completed!"
print_info "═══════════════════════════════════════════════════════════════"

print_info "Results saved to: ${OUTPUT_DIR}/"
print_info ""
print_info "Files created:"
print_info "  - ${RUN_NAME}_full_results.json"
print_info "  - ${RUN_NAME}_summary.json"
print_info "  - qa_results.jsonl"

print_info ""
print_info "Next steps:"
print_info "  1. Review results: cat ${OUTPUT_DIR}/${RUN_NAME}_summary.json"
print_info "  2. Run LongMemEval evaluation:"
print_info "     cd LongMemEval/src/evaluation"
print_info "     python evaluate_qa.py gpt-4o \\"
print_info "         ../../../benchmarks/${OUTPUT_DIR}/qa_results.jsonl \\"
print_info "         ../../../${DATA_DIR}/${DATASET}.json"

print_success "Done!"
