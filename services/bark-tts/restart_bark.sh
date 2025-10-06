#!/bin/bash

# Restart Bark TTS Service Script
echo "🔄 Restarting Bark TTS Service..."

# Kill any existing python processes
echo "🛑 Stopping existing services..."
pkill -f "python3 bark"
pkill -f "bark_service"
pkill -f "bark_final"

# Wait a moment
sleep 2

# Clear any corrupted cache manually
echo "🧹 Clearing model cache..."
rm -rf /root/.local/share/tts/tts_models--multilingual--multi-dataset--bark
rm -rf ~/.local/share/tts/tts_models--multilingual--multi-dataset--bark
rm -rf /tmp/tts_cache

# Set environment variables for better compatibility
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

# Start the service
echo "🚀 Starting Bark TTS service..."
cd /workspace/unmute-02/services/bark-tts
python3 bark_service.py &

echo "✅ Service started! Check logs for status."
echo "🔍 Use 'curl http://localhost:8006/health' to check health"
