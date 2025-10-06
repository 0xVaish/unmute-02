"""
Dia TTS Service - Ultra-realistic dialogue generation
Using Nari Labs Dia model for natural human-like voice synthesis
"""

import os
import logging
import tempfile
import time
from typing import Optional, List
import asyncio

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHECKPOINT = "nari-labs/Dia-1.6B-0626"

# Global model instances
processor = None
model = None

# Create FastAPI app
app = FastAPI(
    title="Dia TTS Service",
    description="Ultra-realistic dialogue generation using Nari Labs Dia",
    version="1.0.0"
)

class TTSRequest(BaseModel):
    text: str
    speaker: Optional[str] = "S1"  # S1 or S2 for dialogue
    guidance_scale: Optional[float] = 3.0
    temperature: Optional[float] = 1.8
    top_p: Optional[float] = 0.90
    top_k: Optional[int] = 45
    max_new_tokens: Optional[int] = 3072

class DialogueRequest(BaseModel):
    dialogue: List[dict]  # [{"speaker": "S1", "text": "Hello"}, {"speaker": "S2", "text": "Hi"}]
    guidance_scale: Optional[float] = 3.0
    temperature: Optional[float] = 1.8
    top_p: Optional[float] = 0.90
    top_k: Optional[int] = 45
    max_new_tokens: Optional[int] = 3072

def format_dialogue_text(dialogue_list: List[dict]) -> str:
    """Format dialogue list into Dia format with [S1] and [S2] tags"""
    formatted_text = ""
    for item in dialogue_list:
        speaker = item.get("speaker", "S1")
        text = item.get("text", "")
        formatted_text += f"[{speaker}] {text} "
    return formatted_text.strip()

def format_single_text(text: str, speaker: str = "S1") -> str:
    """Format single text with speaker tag"""
    return f"[{speaker}] {text}"

@app.on_event("startup")
async def startup_event():
    """Load Dia model on startup"""
    global processor, model
    
    try:
        logger.info(f"🎤 Loading Dia TTS model on {DEVICE}")
        
        # Import transformers
        from transformers import AutoProcessor, DiaForConditionalGeneration
        
        # Load processor and model
        logger.info("📥 Loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT)
        
        logger.info("📥 Loading model...")
        model = DiaForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT).to(DEVICE)
        
        logger.info("✅ Dia TTS model loaded successfully")
        
        # Test the model with a simple synthesis
        test_text = ["[S1] Hello, this is a test of the Dia TTS system."]
        test_inputs = processor(text=test_text, padding=True, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            test_outputs = model.generate(
                **test_inputs,
                max_new_tokens=1024,
                guidance_scale=3.0,
                temperature=1.8,
                top_p=0.90,
                top_k=45
            )
        
        logger.info("🎯 Model test synthesis successful")
        
    except Exception as e:
        logger.error(f"❌ Failed to load Dia TTS model: {e}")
        processor = None
        model = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    if processor is None or model is None:
        return {"status": "degraded", "message": "Model not loaded", "device": DEVICE}
    
    return {
        "status": "healthy",
        "model": MODEL_CHECKPOINT,
        "device": DEVICE,
        "features": ["dialogue_generation", "voice_synthesis", "non_verbal_sounds"]
    }

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text using Dia TTS
    
    Args:
        request: TTS request with text and generation parameters
    
    Returns:
        Audio file as MP3
    """
    
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="Dia TTS not available")
    
    try:
        logger.info(f"🎯 Synthesizing: '{request.text[:50]}...'")
        
        # Format text with speaker tag
        formatted_text = format_single_text(request.text, request.speaker)
        
        # Process input
        inputs = processor(text=[formatted_text], padding=True, return_tensors="pt").to(DEVICE)
        
        # Generate audio
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                guidance_scale=request.guidance_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
        
        # Decode outputs
        decoded_outputs = processor.batch_decode(outputs)
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            processor.save_audio(decoded_outputs, temp_file.name)
            
            # Read back as bytes
            with open(temp_file.name, "rb") as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
        
        logger.info("✅ Speech synthesis complete")
        
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
        
    except Exception as e:
        logger.error(f"❌ Speech synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/dialogue")
async def synthesize_dialogue(request: DialogueRequest):
    """
    Synthesize dialogue between multiple speakers
    
    Args:
        request: Dialogue request with speaker turns and generation parameters
    
    Returns:
        Audio file as MP3
    """
    
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="Dia TTS not available")
    
    try:
        logger.info(f"🎭 Synthesizing dialogue with {len(request.dialogue)} turns")
        
        # Format dialogue
        formatted_text = format_dialogue_text(request.dialogue)
        logger.info(f"📝 Formatted dialogue: {formatted_text[:100]}...")
        
        # Process input
        inputs = processor(text=[formatted_text], padding=True, return_tensors="pt").to(DEVICE)
        
        # Generate audio
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                guidance_scale=request.guidance_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
        
        # Decode outputs
        decoded_outputs = processor.batch_decode(outputs)
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            processor.save_audio(decoded_outputs, temp_file.name)
            
            # Read back as bytes
            with open(temp_file.name, "rb") as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
        
        logger.info("✅ Dialogue synthesis complete")
        
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=dialogue.mp3"}
        )
        
    except Exception as e:
        logger.error(f"❌ Dialogue synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/features")
async def get_features():
    """Get available features and non-verbal sounds"""
    
    return {
        "speakers": ["S1", "S2"],
        "non_verbals": [
            "(laughs)", "(clears throat)", "(sighs)", "(gasps)", "(coughs)",
            "(singing)", "(sings)", "(mumbles)", "(beep)", "(groans)",
            "(sniffs)", "(claps)", "(screams)", "(inhales)", "(exhales)",
            "(applause)", "(burps)", "(humming)", "(sneezes)", "(chuckle)",
            "(whistles)"
        ],
        "generation_parameters": {
            "guidance_scale": {"default": 3.0, "range": [1.0, 10.0]},
            "temperature": {"default": 1.8, "range": [0.1, 2.0]},
            "top_p": {"default": 0.90, "range": [0.1, 1.0]},
            "top_k": {"default": 45, "range": [1, 100]},
            "max_new_tokens": {"default": 3072, "range": [512, 4096]}
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "dia_service:app",
        host="0.0.0.0",
        port=8006,
        log_level="info"
    )
