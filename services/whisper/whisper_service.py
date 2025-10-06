"""
Whisper STT Service
High-quality speech-to-text using OpenAI Whisper
"""

import os
import io
import logging
import tempfile
from typing import Optional

import whisper
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")  # tiny, base, small, medium, large
DEVICE = "cuda" if torch.cuda.is_available() and os.getenv("DEVICE") == "cuda" else "cpu"
LANGUAGE = os.getenv("LANGUAGE", "en")

# Global model instance
whisper_model = None

# Create FastAPI app
app = FastAPI(
    title="Whisper STT Service",
    description="Speech-to-Text using OpenAI Whisper",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load Whisper model on startup"""
    global whisper_model
    
    logger.info(f"🎤 Loading Whisper model: {MODEL_SIZE} on {DEVICE}")
    
    try:
        whisper_model = whisper.load_model(MODEL_SIZE, device=DEVICE)
        logger.info(f"✅ Whisper model loaded successfully")
        
        # Warm up the model
        logger.info("🔥 Warming up model...")
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        whisper_model.transcribe(dummy_audio, language=LANGUAGE)
        logger.info("✅ Model warmed up")
        
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "language": LANGUAGE
    }


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio to text
    
    Args:
        audio: Audio file (wav, mp3, etc.)
    
    Returns:
        JSON with transcript and metadata
    """
    
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe
            logger.info(f"🎯 Transcribing audio ({len(audio_data)} bytes)")
            
            result = whisper_model.transcribe(
                temp_file_path,
                language=LANGUAGE,
                fp16=False,  # Use fp32 for better compatibility
                verbose=False
            )
            
            transcript = result["text"].strip()
            
            logger.info(f"✅ Transcription complete: '{transcript[:50]}...'")
            
            return {
                "transcript": transcript,
                "language": result.get("language", LANGUAGE),
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip()
                    }
                    for seg in result.get("segments", [])
                ],
                "confidence": _calculate_average_confidence(result.get("segments", []))
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/transcribe_streaming")
async def transcribe_streaming(audio: UploadFile = File(...)):
    """
    Transcribe audio with streaming-like response (faster processing)
    
    Args:
        audio: Audio file chunk
    
    Returns:
        JSON with partial transcript
    """
    
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        # For streaming, we use a smaller processing window
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Quick transcription with reduced parameters for speed
            result = whisper_model.transcribe(
                temp_file_path,
                language=LANGUAGE,
                fp16=False,
                verbose=False,
                condition_on_previous_text=False,  # Faster processing
                temperature=0.0  # More deterministic
            )
            
            transcript = result["text"].strip()
            
            return {
                "transcript": transcript,
                "is_partial": len(audio_data) < 32000,  # Less than 2 seconds
                "confidence": _calculate_average_confidence(result.get("segments", []))
            }
            
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Streaming transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/models")
async def get_model_info():
    """Get information about the loaded model"""
    
    return {
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "language": LANGUAGE,
        "available_models": ["tiny", "base", "small", "medium", "large"],
        "model_loaded": whisper_model is not None
    }


def _calculate_average_confidence(segments):
    """Calculate average confidence from segments"""
    
    if not segments:
        return 0.0
    
    # Whisper doesn't provide confidence directly, so we estimate based on segment properties
    total_confidence = 0.0
    total_duration = 0.0
    
    for segment in segments:
        duration = segment.get("end", 0) - segment.get("start", 0)
        
        # Estimate confidence based on segment characteristics
        text_length = len(segment.get("text", "").strip())
        
        # Longer segments with more text typically have higher confidence
        estimated_confidence = min(0.95, 0.6 + (text_length / 100) * 0.3)
        
        total_confidence += estimated_confidence * duration
        total_duration += duration
    
    return total_confidence / total_duration if total_duration > 0 else 0.0


if __name__ == "__main__":
    uvicorn.run(
        "whisper_service:app",
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
