#!/bin/bash

# Restart Dia TTS Service Script
echo "🔄 Restarting Dia TTS Service..."

# Kill any existing python processes
echo "🛑 Stopping existing services..."
pkill -f "python3 dia"
pkill -f "dia_service"
pkill -f "bark"  # Also stop any bark services

# Wait a moment
sleep 2

# Set environment variables for better compatibility
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

# Install dependencies with conflict resolution
echo "📦 Installing Dia TTS with fixed dependencies..."
pip install -r /workspace/unmute-02/services/dia-tts/requirements.txt

# Start the service
echo "🚀 Starting Dia TTS service..."
cd /workspace/unmute-02/services/dia-tts
python3 dia_service.py &

echo "✅ Dia TTS service started!"
echo "🔍 Use 'curl http://localhost:8005/health' to check health"
echo "🎭 Use 'curl http://localhost:8005/features' to see available features"
