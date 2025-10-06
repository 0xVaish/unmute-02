#!/bin/bash

# Install Dia TTS Service Script
echo "🚀 Installing Dia TTS Service..."

# Stop any existing services
echo "🛑 Stopping existing services..."
pkill -f "python3 bark"
pkill -f "python3 dia"
pkill -f "bark_service"
pkill -f "dia_service"

# Wait a moment
sleep 2

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update
apt-get install -y git ffmpeg

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip

# Install main branch of transformers (required for Dia)
echo "🤗 Installing latest transformers..."
pip install git+https://github.com/huggingface/transformers.git

# Install other dependencies
pip install torch torchaudio fastapi uvicorn pydantic numpy soundfile

# Install Dia TTS
echo "🎤 Installing Dia TTS..."
pip install git+https://github.com/nari-labs/dia.git

# Make scripts executable
chmod +x /workspace/unmute-02/services/dia-tts/restart_dia.sh

echo "✅ Dia TTS installation complete!"
echo "🔍 Run './services/dia-tts/restart_dia.sh' to start the service"
