#!/bin/bash

# Fix All Dependencies Properly for Dia TTS
echo "🔧 Fixing all dependency conflicts for Dia TTS..."

# Stop any existing services
echo "🛑 Stopping all services..."
pkill -f python3
sleep 3

# Create a clean virtual environment approach
echo "🧹 Cleaning up conflicting packages..."

# Uninstall conflicting packages first
pip uninstall -y myshell-openvoice gruut tts nari-tts transformers tokenizers gradio numpy protobuf

# Install compatible versions in correct order
echo "📦 Installing compatible dependencies..."

# 1. Install numpy first (base dependency)
pip install numpy==1.22.0

# 2. Install protobuf compatible version
pip install protobuf==3.20.3

# 3. Install tokenizers compatible with faster-whisper
pip install tokenizers==0.13.4

# 4. Install gradio compatible version
pip install gradio==3.48.0

# 5. Install librosa compatible version
pip install librosa==0.9.1

# 6. Install TTS with compatible numpy
pip install TTS==0.22.0

# 7. Install torch with compatible versions
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 8. Install main branch transformers for Dia support
pip install git+https://github.com/huggingface/transformers.git

# 9. Install FastAPI stack
pip install fastapi==0.100.0 uvicorn[standard]==0.22.0 pydantic==2.0.0

# 10. Install audio processing
pip install soundfile==0.12.0

# 11. Finally install Dia TTS
pip install git+https://github.com/nari-labs/dia.git

echo "✅ Testing imports..."
python3 -c "
try:
    from transformers import AutoProcessor, DiaForConditionalGeneration
    print('✅ Transformers with Dia support working')
    import numpy as np
    print('✅ Numpy version:', np.__version__)
    import torch
    print('✅ Torch version:', torch.__version__)
    import TTS
    print('✅ TTS library working')
    print('🎉 All dependencies resolved!')
except Exception as e:
    print('❌ Import error:', e)
"

# Start Dia service
echo "🚀 Starting Dia TTS service..."
cd /workspace/unmute-02/services/dia-tts
python3 dia_service.py &

sleep 10
echo "🔍 Testing service..."
curl http://localhost:8005/health

echo "✅ Dia TTS with fixed dependencies is ready!"
