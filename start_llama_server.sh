#!/bin/bash
# Start llama-server pinned to Tesla P40 GPU (CUDA device 0)
# Model: davidau-qwen38 (Q4_K_M GGUF)
# Port: 9931

set -e

MODEL_PATH="/home/jonathan/Desktop/P5/models/davidau-qwen38-q4_k_m.gguf"
PORT=9931
ALIAS="davidau-qwen38"

echo "Starting llama-server on GPU 0 (Tesla P40)..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Alias: $ALIAS"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

# Kill any existing llama-server on this port
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 1

# Start llama-server pinned to GPU 0
CUDA_VISIBLE_DEVICES=0 llama-server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --alias "$ALIAS" \
    --n-gpu-layers 99 \
    --ctx-size 16384 \
    --batch-size 512 \
    --parallel 4 \
    --log-disable \
    --verbose-prompt

echo "llama-server starting on port $PORT..."

# Wait for server to be ready
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "llama-server is ready!"
        echo "Model alias: $ALIAS"
        echo "Endpoint: http://localhost:$PORT/v1"
        exit 0
    fi
    sleep 1
done

echo "ERROR: llama-server did not start within 30 seconds"
exit 1
