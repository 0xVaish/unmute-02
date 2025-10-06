"""
Bark TTS Service - Enhanced with Better Error Handling
Most natural human-like voice synthesis using TTS library
"""

import os
import io
import logging
import tempfile
import shutil
import time
from typing import Optional

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
MODEL_NAME = "tts_models/multilingual/multi-dataset/bark"

# Global model instance
tts_model = None

# Create FastAPI app    
app = FastAPI(
    title="Bark TTS Service",
    description="Natural human-like voice synthesis",
    version="1.0.0"
)

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "v2/en_speaker_6"
    speed: Optional[float] = 1.0

def clear_corrupted_cache():
    """Clear potentially corrupted model cache"""
    cache_dirs = [
        "/root/.local/share/tts/tts_models--multilingual--multi-dataset--bark",
        os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--bark"),
        "/tmp/tts_cache"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                logger.info(f"🧹 Clearing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)
            except Exception as e:
                logger.warning(f"Could not clear cache {cache_dir}: {e}")

def setup_torch_compatibility():
    """Setup torch for better compatibility"""
    # Environment variables for better compatibility
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Override torch.load for compatibility
    import torch._utils
    original_load = torch.load
    
    def safe_load(*args, **kwargs):
        kwargs.pop('weights_only', None)
        try:
            return original_load(*args, weights_only=False, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load with safe_load: {e}")
            # Try with map_location
            kwargs['map_location'] = 'cpu'
            return original_load(*args, weights_only=False, **kwargs)
    
    torch.load = safe_load
    return original_load

@app.on_event("startup")
async def startup_event():
    """Load TTS model with comprehensive error handling"""
    global tts_model
    
    try:
        # Import TTS library
        from TTS.api import TTS
        
        logger.info(f"🎤 Loading Bark TTS model on {DEVICE}")
        
        # Clear any corrupted cache first
        clear_corrupted_cache()
        
        # Setup torch compatibility
        original_load = setup_torch_compatibility()
        
        # Try loading with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"📥 Attempt {attempt + 1}/{max_retries} to load model...")
                
                # Load model with explicit device handling
                tts_model = TTS(MODEL_NAME)
                
                if DEVICE == "cuda" and torch.cuda.is_available():
                    tts_model = tts_model.to(DEVICE)
                    logger.info(f"✅ Bark model loaded on {DEVICE}")
                else:
                    logger.info("✅ Bark model loaded on CPU")
                
                # Test the model with a simple synthesis
                test_audio = tts_model.tts("Hello")
                logger.info("🎯 Model test synthesis successful")
                break
                
            except Exception as e:
                logger.error(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info("🔄 Clearing cache and retrying...")
                    clear_corrupted_cache()
                    time.sleep(5)  # Wait before retry
                else:
                    logger.error("❌ All attempts failed")
                    tts_model = None
        
        # Restore original torch.load
        torch.load = original_load
        
        if tts_model is not None:
            logger.info("✅ Bark TTS service ready")
        else:
            logger.info("🔄 Service will run in degraded mode")
            
    except ImportError:
        logger.error("❌ TTS library is not available")
    except Exception as e:
        logger.error(f"❌ Failed to load Bark TTS model: {e}")
        logger.info("🔄 Service will run in degraded mode")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    if tts_model is None:
        return {"status": "degraded", "message": "Model not loaded", "device": DEVICE}
    
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "tts_available": True
    }

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text
    
    Args:
        request: TTS request with text and voice parameters
    
    Returns:
        Audio file as WAV
    """
    
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Bark TTS not available")
    
    try:
        logger.info(f"🎯 Synthesizing: '{request.text[:50]}...'")
        
        # Generate audio using TTS
        audio = tts_model.tts(
            text=request.text,
            speaker=request.voice if hasattr(tts_model, 'speakers') else None
        )
        
        # Convert to numpy array if needed
        if isinstance(audio, list):
            audio = np.array(audio)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            sf.write(temp_file.name, audio, 24000)  # Bark uses 24kHz
            
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

@app.get("/voices")
async def get_available_voices():
    """Get available voice options"""
    
    return {
        "voices": [
            "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
            "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
            "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8",
            "v2/en_speaker_9"
        ],
        "default": "v2/en_speaker_6"
    }

if __name__ == "__main__":
    uvicorn.run(
        "bark_service:app",
        host="0.0.0.0",
        port=8006,
        log_level="info"
    )
