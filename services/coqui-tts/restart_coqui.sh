#!/bin/bash

# Restart Coqui TTS Service Script
echo "🔄 Starting Coqui TTS Service..."

# Kill any existing python processes
echo "🛑 Stopping existing services..."
pkill -f "python3"
sleep 3

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update
apt-get install -y espeak-ng ffmpeg

# Install Coqui TTS with compatible versions
echo "🎤 Installing Coqui TTS..."
pip install TTS==0.22.0
pip install numpy==1.22.0  # Compatible with TTS
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Start the service
echo "🚀 Starting Coqui TTS service..."
cd /workspace/unmute-02/services/coqui-tts
python3 coqui_service.py &

echo "✅ Coqui TTS service started!"
echo "🔍 Use 'curl http://localhost:8005/health' to check health"
echo "🎭 Use 'curl http://localhost:8005/voices' to see available voices"
