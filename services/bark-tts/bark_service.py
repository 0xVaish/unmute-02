"""
Bark TTS Service
Most natural human-like voice synthesis using Bark
"""

import os
import io
import logging
import tempfile
from typing import Optional

import torch
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# Import Bark
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from bark.generation import set_seed
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    logging.warning("Bark not available, using fallback")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() and os.getenv("DEVICE") == "cuda" else "cpu"
VOICE_PRESET = os.getenv("VOICE_PRESET", "v2/en_speaker_6")  # Professional female
VOICE_PRESET_MALE = os.getenv("VOICE_PRESET_MALE", "v2/en_speaker_9")  # Professional male
ENABLE_VOICE_CLONING = os.getenv("ENABLE_VOICE_CLONING", "true").lower() == "true"

# Model loaded flag
models_loaded = False

# Create FastAPI app
app = FastAPI(
    title="Bark TTS Service",
    description="Natural human-like voice synthesis using Bark",
    version="1.0.0"
)


class TTSRequest(BaseModel):
    text: str
    voice_preset: Optional[str] = None
    voice_sample: Optional[str] = None
    temperature: float = 0.7
    silence_padding: float = 0.25


@app.on_event("startup")
async def startup_event():
    """Load Bark models on startup"""
    global models_loaded
    
    if not BARK_AVAILABLE:
        logger.error("❌ Bark is not available")
        return
    
    logger.info(f"🎵 Loading Bark models on {DEVICE}")
    
    try:
        # Preload all Bark models
        preload_models(
            text_use_gpu=DEVICE == "cuda",
            text_use_small=False,
            coarse_use_gpu=DEVICE == "cuda",
            coarse_use_small=False,
            fine_use_gpu=DEVICE == "cuda",
            fine_use_small=False,
            codec_use_gpu=DEVICE == "cuda"
        )
        
        models_loaded = True
        logger.info("✅ Bark models loaded successfully")
        
        # Warm up with a short generation
        logger.info("🔥 Warming up Bark...")
        set_seed(42)
        _ = generate_audio("Hello", history_prompt=VOICE_PRESET)
        logger.info("✅ Bark warmed up")
        
    except Exception as e:
        logger.error(f"❌ Failed to load Bark models: {e}")
        models_loaded = False


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    if not BARK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Bark not available")
    
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "device": DEVICE,
        "voice_preset": VOICE_PRESET,
        "voice_cloning_enabled": ENABLE_VOICE_CLONING,
        "sample_rate": SAMPLE_RATE
    }


@app.post("/generate")
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text using Bark
    
    Args:
        request: TTS request with text and voice parameters
    
    Returns:
        Audio file (WAV format)
    """
    
    if not BARK_AVAILABLE or not models_loaded:
        raise HTTPException(status_code=503, detail="Bark service not available")
    
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(text) > 500:
            raise HTTPException(status_code=400, detail="Text too long (max 500 characters)")
        
        logger.info(f"🎵 Generating speech for: '{text[:50]}...'")
        
        # Select voice preset
        voice_preset = request.voice_preset or VOICE_PRESET
        
        # Set random seed for consistency
        set_seed(42)
        
        # Handle voice cloning if provided
        if request.voice_sample and ENABLE_VOICE_CLONING:
            # TODO: Implement voice cloning from sample
            # For now, use default preset
            logger.info("🎭 Voice cloning requested (using default preset for now)")
        
        # Generate audio
        audio_array = generate_audio(
            text,
            history_prompt=voice_preset,
            text_temp=request.temperature,
            waveform_temp=request.temperature
        )
        
        # Add silence padding if requested
        if request.silence_padding > 0:
            silence_samples = int(SAMPLE_RATE * request.silence_padding)
            silence = np.zeros(silence_samples, dtype=audio_array.dtype)
            audio_array = np.concatenate([silence, audio_array, silence])
        
        # Convert to WAV format
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_array, SAMPLE_RATE, format='WAV')
        audio_data = audio_buffer.getvalue()
        
        logger.info(f"✅ Generated {len(audio_data)} bytes of audio")
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=bark_output.wav",
                "X-Audio-Duration": str(len(audio_array) / SAMPLE_RATE),
                "X-Sample-Rate": str(SAMPLE_RATE)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Speech generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")


@app.post("/generate_with_emotions")
async def generate_speech_with_emotions(request: TTSRequest):
    """
    Generate speech with emotional markers
    
    Bark supports emotional cues in text like:
    - [laughs], [sighs], [clears throat]
    - CAPITALIZATION for emphasis
    - ... for pauses
    """
    
    if not BARK_AVAILABLE or not models_loaded:
        raise HTTPException(status_code=503, detail="Bark service not available")
    
    try:
        text = request.text.strip()
        
        # Add some natural speech patterns for medical context
        if "hello" in text.lower() and "assistant" in text.lower():
            text = f"[clears throat] {text}"
        
        # Use the regular generation with emotional text
        request.text = text
        return await generate_speech(request)
        
    except Exception as e:
        logger.error(f"❌ Emotional speech generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")


@app.get("/voices")
async def get_available_voices():
    """Get list of available voice presets"""
    
    # Bark voice presets (professional medical voices)
    voices = {
        "female_professional": {
            "id": "v2/en_speaker_6",
            "name": "Dr. Sarah",
            "gender": "female",
            "description": "Professional, warm female doctor voice",
            "best_for": ["consultations", "explanations", "empathetic responses"]
        },
        "male_professional": {
            "id": "v2/en_speaker_9", 
            "name": "Dr. Michael",
            "gender": "male",
            "description": "Calm, authoritative male doctor voice",
            "best_for": ["emergency guidance", "serious discussions"]
        },
        "female_friendly": {
            "id": "v2/en_speaker_0",
            "name": "Nurse Jenny",
            "gender": "female", 
            "description": "Gentle, caring female voice",
            "best_for": ["patient care", "comfort", "medication reminders"]
        },
        "neutral_assistant": {
            "id": "v2/en_speaker_3",
            "name": "Medical Assistant",
            "gender": "neutral",
            "description": "Professional, clear assistant voice",
            "best_for": ["information", "appointments", "general queries"]
        }
    }
    
    return {
        "voices": voices,
        "default_voice": VOICE_PRESET,
        "voice_cloning_enabled": ENABLE_VOICE_CLONING,
        "emotional_markers": [
            "[laughs]", "[sighs]", "[clears throat]", 
            "[whispers]", "[shouting]", "..."
        ]
    }


@app.post("/test_voice")
async def test_voice(voice_preset: str = "v2/en_speaker_6"):
    """
    Test a voice preset with a standard medical phrase
    """
    
    test_text = "Hello, I'm your medical assistant. How can I help you today?"
    
    request = TTSRequest(
        text=test_text,
        voice_preset=voice_preset,
        temperature=0.7
    )
    
    return await generate_speech(request)


@app.get("/model_info")
async def get_model_info():
    """Get information about the Bark model"""
    
    return {
        "model_name": "Bark",
        "version": "1.0",
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE if BARK_AVAILABLE else None,
        "max_text_length": 500,
        "voice_cloning": ENABLE_VOICE_CLONING,
        "emotional_synthesis": True,
        "languages": ["English"],
        "model_loaded": models_loaded,
        "bark_available": BARK_AVAILABLE
    }


if __name__ == "__main__":
    uvicorn.run(
        "bark_service:app",
        host="0.0.0.0",
        port=8005,
        log_level="info"
    )
