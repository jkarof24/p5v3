#!/bin/bash
# Run P5 experiments on CPU (no GPU)
# Uses llama-server at localhost:9931 for LLM inference
# All PyTorch computation runs on CPU

set -e

PYTHON="/home/jonathan/venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "P5 Experiment Runner"
echo "===================="
echo "Python: $PYTHON"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Force CPU-only for PyTorch
export CUDA_VISIBLE_DEVICES=""

# Check if llama-server is running
if ! curl -s "http://localhost:9931/v1/models" > /dev/null 2>&1; then
    echo "WARNING: llama-server not running on localhost:9931"
    echo "Start it with: ./start_llama_server.sh"
    echo ""
fi

# Run experiments
if [ "${1:-}" = "--dry_run" ]; then
    echo "Running in dry-run mode..."
    $PYTHON run_experiments.py --dry_run "$@"
else
    echo "Running experiments..."
    $PYTHON run_experiments.py "$@"
fi

echo ""
echo "Experiments completed!"
echo "Results saved to results/ directory"
