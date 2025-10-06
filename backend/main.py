"""
Medical Voice Assistant Backend
Ultra Human-Like Voice with Smart Selection
- Higgs Audio V2 (Best Quality)
- Chatterbox (Fast Backup) 
- Kokoro (Real-time)
- Whisper STT + Gemini LLM
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import settings
from core.smart_voice_engine import SmartVoiceEngine
from core.medical_assistant import MedicalAssistant
from core.session_manager import SessionManager
from utils.logger import setup_logger
from utils.metrics import MetricsCollector

# Setup logging
logger = setup_logger(__name__)

# Global instances
voice_engine: Optional[SmartVoiceEngine] = None
medical_assistant: Optional[MedicalAssistant] = None
session_manager: Optional[SessionManager] = None
metrics: Optional[MetricsCollector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global voice_engine, medical_assistant, session_manager, metrics
    
    logger.info("🏥 Starting Medical Voice Assistant with Ultra Human-Like Voice")
    
    try:
        # Initialize metrics
        metrics = MetricsCollector()
        logger.info("✅ Metrics initialized")
        
        # Initialize Smart Voice Engine (Multi-TTS)
        voice_engine = SmartVoiceEngine(
            whisper_url=settings.WHISPER_URL,
            higgs_url=settings.HIGGS_URL,
            chatterbox_url=settings.CHATTERBOX_URL,
            kokoro_url=settings.KOKORO_URL,
            max_latency_ms=settings.MAX_LATENCY_MS
        )
        await voice_engine.initialize()
        logger.info("✅ Smart Voice Engine initialized (4 TTS models)")
        
        # Initialize Medical Assistant with Gemini
        medical_assistant = MedicalAssistant(
            use_gemini=settings.USE_GEMINI,
            api_key=settings.GOOGLE_API_KEY
        )
        await medical_assistant.initialize()
        logger.info("✅ Medical Assistant initialized (Gemini)")
        
        # Initialize Session Manager
        session_manager = SessionManager(
            redis_url=settings.REDIS_URL,
            max_sessions=settings.MAX_CONCURRENT_SESSIONS
        )
        await session_manager.initialize()
        logger.info("✅ Session Manager initialized")
        
        logger.info("🚀 Medical Voice Assistant Ready!")
        logger.info(f"   - Primary TTS: {settings.PRIMARY_TTS}")
        logger.info(f"   - Fallback TTS: {settings.FALLBACK_TTS}")
        logger.info(f"   - Real-time TTS: {settings.REALTIME_TTS}")
        logger.info(f"   - Max Latency: {settings.MAX_LATENCY_MS}ms")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    finally:
        # Cleanup
        if session_manager:
            await session_manager.cleanup()
        if voice_engine:
            await voice_engine.cleanup()
        if medical_assistant:
            await medical_assistant.cleanup()
        logger.info("🔄 Cleanup completed")


# Create FastAPI app
app = FastAPI(
    title="Medical Voice Assistant - Ultra Human Voice",
    description="Real-time speech-to-speech with the most human-like voices",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class VoiceQualityRequest(BaseModel):
    text: str
    voice_quality: str = "best"  # best, fast, realtime
    voice_clone_sample: Optional[str] = None


class ConversationResponse(BaseModel):
    session_id: str
    transcript: str
    response_text: str
    audio_url: Optional[str] = None
    voice_model_used: str
    latency_ms: int
    confidence_score: float
    medical_flags: List[str] = []


# Health check
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        services_status = {}
        
        if voice_engine:
            # Check all TTS services
            higgs_status = await voice_engine.check_higgs_health()
            chatterbox_status = await voice_engine.check_chatterbox_health()
            kokoro_status = await voice_engine.check_kokoro_health()
            whisper_status = await voice_engine.check_whisper_health()
            
            services_status.update({
                "higgs_tts": higgs_status,
                "chatterbox_tts": chatterbox_status,
                "kokoro_tts": kokoro_status,
                "whisper_stt": whisper_status
            })
        
        if medical_assistant:
            services_status["gemini_llm"] = await medical_assistant.health_check()
        
        if session_manager:
            services_status["session_manager"] = await session_manager.health_check()
        
        # Overall status
        all_healthy = all(status == "healthy" for status in services_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": services_status,
            "voice_models": {
                "primary": settings.PRIMARY_TTS,
                "fallback": settings.FALLBACK_TTS,
                "realtime": settings.REALTIME_TTS
            },
            "performance": {
                "max_latency_ms": settings.MAX_LATENCY_MS,
                "auto_fallback": settings.AUTO_FALLBACK
            },
            "uptime": metrics.get_uptime() if metrics else 0.0
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# Voice quality demo endpoint
@app.post("/api/voice/demo")
async def voice_demo(request: VoiceQualityRequest):
    """Demo different voice qualities"""
    try:
        if not voice_engine:
            raise HTTPException(status_code=503, detail="Voice engine not available")
        
        start_time = time.time()
        
        # Select TTS based on quality preference
        if request.voice_quality == "best":
            audio_data, model_used = await voice_engine.generate_with_higgs(
                request.text, 
                voice_sample=request.voice_clone_sample
            )
        elif request.voice_quality == "fast":
            audio_data, model_used = await voice_engine.generate_with_chatterbox(
                request.text,
                voice_sample=request.voice_clone_sample
            )
        elif request.voice_quality == "realtime":
            audio_data, model_used = await voice_engine.generate_with_kokoro(
                request.text
            )
        else:
            # Smart selection based on latency
            audio_data, model_used = await voice_engine.smart_generate(
                request.text,
                voice_sample=request.voice_clone_sample
            )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "audio_data": audio_data,
            "model_used": model_used,
            "latency_ms": latency_ms,
            "text": request.text,
            "quality_level": request.voice_quality
        }
        
    except Exception as e:
        logger.error(f"Voice demo failed: {e}")
        raise HTTPException(status_code=500, detail="Voice generation failed")


# WebSocket for real-time conversation
@app.websocket("/ws/{session_id}")
async def websocket_conversation(websocket: WebSocket, session_id: str):
    """Real-time voice conversation with smart TTS selection"""
    await websocket.accept()
    logger.info(f"🔌 WebSocket connected: {session_id}")
    
    if not all([voice_engine, medical_assistant, session_manager]):
        await websocket.close(code=1011, reason="Services not initialized")
        return
    
    try:
        # Create session
        session = await session_manager.create_session(session_id)
        
        # Send initial greeting
        greeting = await medical_assistant.get_greeting()
        greeting_audio, model_used = await voice_engine.smart_generate(greeting)
        
        await websocket.send_json({
            "type": "greeting",
            "text": greeting,
            "audio_data": greeting_audio,
            "model_used": model_used
        })
        
        # Conversation loop
        async for message in websocket.iter_bytes():
            try:
                conversation_start = time.time()
                
                # Speech to Text
                transcript = await voice_engine.speech_to_text(message)
                
                if transcript.strip():
                    logger.info(f"📝 User: {transcript}")
                    
                    # Get medical response
                    response = await medical_assistant.process_message(
                        transcript, 
                        session_id,
                        session.context
                    )
                    
                    # Smart TTS selection based on context
                    if any(flag in response.medical_flags for flag in ["emergency", "urgent"]):
                        # Use fastest TTS for emergencies
                        audio_response, tts_model = await voice_engine.generate_with_kokoro(
                            response.response_text
                        )
                    elif len(response.response_text) > 200:
                        # Use best quality for long responses
                        audio_response, tts_model = await voice_engine.generate_with_higgs(
                            response.response_text,
                            voice_sample=session.voice_sample
                        )
                    else:
                        # Smart selection for normal responses
                        audio_response, tts_model = await voice_engine.smart_generate(
                            response.response_text,
                            voice_sample=session.voice_sample
                        )
                    
                    total_latency = int((time.time() - conversation_start) * 1000)
                    
                    # Update session
                    await session_manager.update_session(session_id, {
                        "last_transcript": transcript,
                        "last_response": response.response_text,
                        "last_latency": total_latency,
                        "tts_model_used": tts_model
                    })
                    
                    # Send response
                    await websocket.send_json({
                        "type": "conversation_response",
                        "transcript": transcript,
                        "response_text": response.response_text,
                        "audio_data": audio_response,
                        "model_used": tts_model,
                        "latency_ms": total_latency,
                        "medical_flags": response.medical_flags,
                        "confidence_score": response.confidence_score
                    })
                    
                    # Record metrics
                    if metrics:
                        metrics.record_conversation_turn(
                            session_id, 
                            len(transcript), 
                            len(response.response_text),
                            total_latency,
                            tts_model
                        )
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to process audio"
                })
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if session_manager:
            await session_manager.end_session(session_id)


# Get voice models info
@app.get("/api/voice/models")
async def get_voice_models():
    """Get information about available voice models"""
    return {
        "models": {
            "higgs": {
                "name": "Higgs Audio V2",
                "quality": "★★★★★",
                "speed": "Medium",
                "voice_cloning": True,
                "best_for": "Consultations, detailed explanations",
                "avg_latency_ms": 450
            },
            "chatterbox": {
                "name": "Chatterbox",
                "quality": "★★★★",
                "speed": "Fast",
                "voice_cloning": True,
                "best_for": "Quick responses, general conversation",
                "avg_latency_ms": 280
            },
            "kokoro": {
                "name": "Kokoro",
                "quality": "★★★",
                "speed": "Ultra-Fast",
                "voice_cloning": False,
                "best_for": "Real-time, emergency responses",
                "avg_latency_ms": 120
            }
        },
        "selection_strategy": {
            "primary": settings.PRIMARY_TTS,
            "fallback": settings.FALLBACK_TTS,
            "realtime": settings.REALTIME_TTS,
            "auto_fallback": settings.AUTO_FALLBACK,
            "max_latency_ms": settings.MAX_LATENCY_MS
        }
    }


# Performance metrics
@app.get("/api/metrics")
async def get_metrics():
    """Get performance metrics"""
    if not metrics:
        raise HTTPException(status_code=503, detail="Metrics not available")
    
    return metrics.get_detailed_metrics()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
