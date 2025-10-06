"""
Coqui TTS Service - Stable and High-Quality Text-to-Speech
Using Coqui TTS for reliable voice synthesis in medical applications
"""

import os
import logging
import tempfile
import time
from typing import Optional, List
import asyncio

import torch
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global TTS instance
tts = None

# Create FastAPI app
app = FastAPI(
    title="Coqui TTS Service",
    description="High-quality text-to-speech using Coqui TTS",
    version="1.0.0"
)

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "female"  # female, male
    speed: Optional[float] = 1.0
    emotion: Optional[str] = "neutral"  # neutral, happy, sad, angry

class VoiceCloneRequest(BaseModel):
    text: str
    speaker_wav_path: Optional[str] = None
    language: Optional[str] = "en"

@app.on_event("startup")
async def startup_event():
    """Load Coqui TTS model on startup"""
    global tts
    
    try:
        logger.info(f"🎤 Loading Coqui TTS model on {DEVICE}")
        
        # Import TTS
        from TTS.api import TTS
        
        # Use a stable multilingual model
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        logger.info(f"📥 Loading model: {model_name}")
        tts = TTS(model_name)
        
        if DEVICE == "cuda" and torch.cuda.is_available():
            tts = tts.to(DEVICE)
            logger.info(f"✅ Coqui TTS model loaded on {DEVICE}")
        else:
            logger.info("✅ Coqui TTS model loaded on CPU")
        
        # Test synthesis
        logger.info("🎯 Testing model with sample text...")
        test_audio = tts.tts("Hello, this is a test.", language="en")
        logger.info("✅ Model test successful")
        
    except Exception as e:
        logger.error(f"❌ Failed to load Coqui TTS model: {e}")
        tts = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    if tts is None:
        return {"status": "degraded", "message": "Model not loaded", "device": DEVICE}
    
    return {
        "status": "healthy",
        "model": "xtts_v2",
        "device": DEVICE,
        "features": ["multilingual", "voice_cloning", "emotion_control"]
    }

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text using Coqui TTS
    
    Args:
        request: TTS request with text and voice parameters
    
    Returns:
        Audio file as WAV
    """
    
    if tts is None:
        raise HTTPException(status_code=503, detail="Coqui TTS not available")
    
    try:
        logger.info(f"🎯 Synthesizing: '{request.text[:50]}...'")
        
        # Generate audio
        audio = tts.tts(
            text=request.text,
            language="en"
        )
        
        # Convert to numpy array if needed
        if isinstance(audio, list):
            audio = np.array(audio)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            sf.write(temp_file.name, audio, 22050)  # Coqui uses 22kHz
            
            # Read back as bytes
            with open(temp_file.name, "rb") as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
        
        logger.info("✅ Speech synthesis complete")
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"❌ Speech synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/clone_voice")
async def clone_voice_synthesis(request: VoiceCloneRequest):
    """
    Synthesize speech with voice cloning
    
    Args:
        request: Voice cloning request with text and reference audio
    
    Returns:
        Audio file as WAV
    """
    
    if tts is None:
        raise HTTPException(status_code=503, detail="Coqui TTS not available")
    
    try:
        logger.info(f"🎭 Voice cloning synthesis: '{request.text[:50]}...'")
        
        if request.speaker_wav_path and os.path.exists(request.speaker_wav_path):
            # Voice cloning with reference audio
            audio = tts.tts(
                text=request.text,
                speaker_wav=request.speaker_wav_path,
                language=request.language
            )
        else:
            # Standard synthesis
            audio = tts.tts(
                text=request.text,
                language=request.language
            )
        
        # Convert to numpy array if needed
        if isinstance(audio, list):
            audio = np.array(audio)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            sf.write(temp_file.name, audio, 22050)
            
            # Read back as bytes
            with open(temp_file.name, "rb") as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
        
        logger.info("✅ Voice cloning synthesis complete")
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=cloned_speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"❌ Voice cloning synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/voices")
async def get_available_voices():
    """Get available voice options"""
    
    return {
        "voices": ["female", "male"],
        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja"],
        "emotions": ["neutral", "happy", "sad", "angry"],
        "features": ["voice_cloning", "multilingual", "emotion_control"]
    }

@app.get("/models")
async def get_model_info():
    """Get model information"""
    
    return {
        "current_model": "xtts_v2",
        "model_info": {
            "name": "XTTS v2",
            "description": "Multilingual text-to-speech with voice cloning",
            "languages": 17,
            "voice_cloning": True,
            "real_time": True
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "coqui_service:app",
        host="0.0.0.0",
        port=8005,
        log_level="info"
    )
