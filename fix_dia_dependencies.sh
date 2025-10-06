#!/bin/bash

# Fix Dia TTS Dependencies Script
echo "🔧 Fixing Dia TTS dependencies..."

# Stop any existing services
echo "🛑 Stopping all services..."
pkill -f python3
sleep 3

# Install the main branch of transformers (required for Dia)
echo "🤗 Installing latest transformers from main branch..."
pip uninstall -y transformers
pip install git+https://github.com/huggingface/transformers.git

# Install other required dependencies
echo "📦 Installing additional dependencies..."
pip install --upgrade torch torchaudio
pip install --upgrade fastapi uvicorn pydantic
pip install --upgrade numpy soundfile

# Install Dia TTS fresh
echo "🎤 Installing Dia TTS..."
pip uninstall -y nari-tts
pip install --upgrade git+https://github.com/nari-labs/dia.git

# Verify installations
echo "✅ Verifying installations..."
python3 -c "from transformers import AutoProcessor, DiaForConditionalGeneration; print('✅ Transformers with Dia support installed')"

# Start the service
echo "🚀 Starting Dia TTS service on port 8005..."
cd /workspace/unmute-02/services/dia-tts
python3 dia_service.py &

sleep 5
echo "🔍 Testing service..."
curl http://localhost:8005/health

echo "✅ Dia TTS setup complete!"
