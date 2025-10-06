"""
Smart Voice Engine - Multi-TTS Architecture
Intelligently selects the best TTS model based on context and performance
- Higgs Audio V2: Best quality, emotional
- Chatterbox: Fast, natural, voice cloning
- Bark: Most natural/human-like
- Kokoro: Ultra-fast, real-time
"""

import asyncio
import logging
import time
import httpx
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from core.config import settings, VOICE_MODELS, VOICE_SELECTION_RULES

logger = logging.getLogger(__name__)


class TTSModel(Enum):
    HIGGS = "higgs"
    CHATTERBOX = "chatterbox"
    BARK = "bark"
    KOKORO = "kokoro"


@dataclass
class VoiceResponse:
    audio_data: bytes
    model_used: str
    latency_ms: int
    quality_score: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class ModelHealth:
    model: TTSModel
    is_healthy: bool
    avg_latency_ms: int
    error_rate: float
    last_check: float


class SmartVoiceEngine:
    """
    Smart Voice Engine that manages multiple TTS models and selects the best one
    based on context, performance, and availability
    """
    
    def __init__(self, whisper_url: str, higgs_url: str, chatterbox_url: str, 
                 kokoro_url: str, bark_url: str, max_latency_ms: int = 200):
        self.whisper_url = whisper_url
        self.higgs_url = higgs_url
        self.chatterbox_url = chatterbox_url
        self.kokoro_url = kokoro_url
        self.bark_url = bark_url
        self.max_latency_ms = max_latency_ms
        
        # Model health tracking
        self.model_health: Dict[TTSModel, ModelHealth] = {}
        self.performance_history: Dict[TTSModel, List[float]] = {
            model: [] for model in TTSModel
        }
        
        # HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Selection strategy
        self.primary_model = TTSModel(settings.PRIMARY_TTS)
        self.fallback_model = TTSModel(settings.FALLBACK_TTS)
        self.realtime_model = TTSModel(settings.REALTIME_TTS)
        self.natural_model = TTSModel(settings.NATURAL_TTS)
        
    async def initialize(self):
        """Initialize all TTS services and check their health"""
        logger.info("🎙️ Initializing Smart Voice Engine with 4 TTS models...")
        
        # Initialize model health tracking
        for model in TTSModel:
            self.model_health[model] = ModelHealth(
                model=model,
                is_healthy=False,
                avg_latency_ms=0,
                error_rate=0.0,
                last_check=0.0
            )
        
        # Check health of all models
        await self._check_all_models_health()
        
        # Start background health monitoring
        asyncio.create_task(self._health_monitor_loop())
        
        logger.info("✅ Smart Voice Engine initialized")
        logger.info(f"   Primary: {self.primary_model.value}")
        logger.info(f"   Fallback: {self.fallback_model.value}")
        logger.info(f"   Real-time: {self.realtime_model.value}")
        logger.info(f"   Natural: {self.natural_model.value}")
    
    async def speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech to text using Whisper"""
        try:
            files = {"audio": ("audio.wav", audio_data, "audio/wav")}
            response = await self.client.post(f"{self.whisper_url}/transcribe", files=files)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("transcript", "")
            else:
                logger.error(f"Whisper STT failed: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Speech-to-text error: {e}")
            return ""
    
    async def smart_generate(self, text: str, voice_sample: Optional[str] = None,
                           context: str = "general", priority: str = "balanced") -> Tuple[bytes, str]:
        """
        Intelligently select and use the best TTS model based on context and performance
        
        Args:
            text: Text to synthesize
            voice_sample: Optional voice cloning sample
            context: Context type (emergency, consultation, quick_question, etc.)
            priority: Priority (speed, quality, balanced)
        
        Returns:
            Tuple of (audio_data, model_used)
        """
        
        # Determine best model based on context
        selected_model = self._select_best_model(text, context, priority, voice_sample)
        
        logger.info(f"🎯 Smart selection: {selected_model.value} for context '{context}'")
        
        # Try primary selection
        result = await self._generate_with_model(selected_model, text, voice_sample)
        
        if result.success:
            self._record_performance(selected_model, result.latency_ms, True)
            return result.audio_data, result.model_used
        
        # Fallback strategy
        logger.warning(f"Primary model {selected_model.value} failed, trying fallback...")
        
        fallback_models = self._get_fallback_sequence(selected_model)
        
        for fallback_model in fallback_models:
            if self.model_health[fallback_model].is_healthy:
                result = await self._generate_with_model(fallback_model, text, voice_sample)
                
                if result.success:
                    self._record_performance(fallback_model, result.latency_ms, True)
                    logger.info(f"✅ Fallback successful with {fallback_model.value}")
                    return result.audio_data, result.model_used
        
        # If all models fail, return error
        logger.error("❌ All TTS models failed")
        raise Exception("All TTS models are unavailable")
    
    def _select_best_model(self, text: str, context: str, priority: str, 
                          voice_sample: Optional[str]) -> TTSModel:
        """Select the best TTS model based on various factors"""
        
        # Context-based selection
        if context in VOICE_SELECTION_RULES:
            rule = VOICE_SELECTION_RULES[context]
            preferred_model = TTSModel(rule["preferred_model"])
            
            # Check if preferred model is healthy and meets latency requirements
            if (self.model_health[preferred_model].is_healthy and 
                self.model_health[preferred_model].avg_latency_ms <= rule["max_latency_ms"]):
                return preferred_model
        
        # Priority-based selection
        if priority == "speed":
            # Prefer fastest available model
            for model in [self.realtime_model, self.fallback_model, self.primary_model, self.natural_model]:
                if self.model_health[model].is_healthy:
                    return model
        
        elif priority == "quality":
            # Prefer highest quality available model
            for model in [self.natural_model, self.primary_model, self.fallback_model, self.realtime_model]:
                if self.model_health[model].is_healthy:
                    return model
        
        elif priority == "natural":
            # Prefer most natural sounding model
            for model in [self.natural_model, self.primary_model, self.fallback_model, self.realtime_model]:
                if self.model_health[model].is_healthy:
                    return model
        
        # Default balanced selection
        text_length = len(text)
        
        # For short text, use fast model
        if text_length < 50:
            if self.model_health[self.realtime_model].is_healthy:
                return self.realtime_model
        
        # For medium text, use balanced model
        elif text_length < 200:
            if self.model_health[self.fallback_model].is_healthy:
                return self.fallback_model
        
        # For long text, use best quality model
        else:
            if self.model_health[self.natural_model].is_healthy:
                return self.natural_model
            elif self.model_health[self.primary_model].is_healthy:
                return self.primary_model
        
        # Fallback to any healthy model
        for model in TTSModel:
            if self.model_health[model].is_healthy:
                return model
        
        # If no models are healthy, return primary (will fail gracefully)
        return self.primary_model
    
    def _get_fallback_sequence(self, failed_model: TTSModel) -> List[TTSModel]:
        """Get fallback sequence when a model fails"""
        all_models = [self.natural_model, self.primary_model, self.fallback_model, self.realtime_model]
        
        # Remove the failed model and return the rest in priority order
        fallback_sequence = [model for model in all_models if model != failed_model]
        
        return fallback_sequence
    
    async def _generate_with_model(self, model: TTSModel, text: str, 
                                 voice_sample: Optional[str] = None) -> VoiceResponse:
        """Generate audio with a specific TTS model"""
        
        start_time = time.time()
        
        try:
            if model == TTSModel.HIGGS:
                audio_data = await self._call_higgs_tts(text, voice_sample)
            elif model == TTSModel.CHATTERBOX:
                audio_data = await self._call_chatterbox_tts(text, voice_sample)
            elif model == TTSModel.BARK:
                audio_data = await self._call_bark_tts(text, voice_sample)
            elif model == TTSModel.KOKORO:
                audio_data = await self._call_kokoro_tts(text)
            else:
                raise ValueError(f"Unknown model: {model}")
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return VoiceResponse(
                audio_data=audio_data,
                model_used=model.value,
                latency_ms=latency_ms,
                quality_score=VOICE_MODELS[model.value]["quality_score"],
                success=True
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"TTS generation failed with {model.value}: {e}")
            
            return VoiceResponse(
                audio_data=b"",
                model_used=model.value,
                latency_ms=latency_ms,
                quality_score=0,
                success=False,
                error_message=str(e)
            )
    
    async def _call_higgs_tts(self, text: str, voice_sample: Optional[str] = None) -> bytes:
        """Call Higgs Audio V2 TTS service"""
        payload = {
            "text": text,
            "voice_sample": voice_sample,
            "emotion": "professional_warm"
        }
        
        response = await self.client.post(f"{self.higgs_url}/generate", json=payload)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Higgs TTS failed: {response.status_code}")
    
    async def _call_chatterbox_tts(self, text: str, voice_sample: Optional[str] = None) -> bytes:
        """Call Chatterbox TTS service"""
        payload = {
            "text": text,
            "voice_sample": voice_sample,
            "exaggeration": 0.5,
            "cfg_weight": 0.5
        }
        
        response = await self.client.post(f"{self.chatterbox_url}/generate", json=payload)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Chatterbox TTS failed: {response.status_code}")
    
    async def _call_bark_tts(self, text: str, voice_sample: Optional[str] = None) -> bytes:
        """Call Bark TTS service"""
        payload = {
            "text": text,
            "voice_preset": "v2/en_speaker_6",  # Professional female
            "voice_sample": voice_sample
        }
        
        response = await self.client.post(f"{self.bark_url}/generate", json=payload)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Bark TTS failed: {response.status_code}")
    
    async def _call_kokoro_tts(self, text: str) -> bytes:
        """Call Kokoro TTS service (no voice cloning)"""
        payload = {"text": text}
        
        response = await self.client.post(f"{self.kokoro_url}/generate", json=payload)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Kokoro TTS failed: {response.status_code}")
    
    async def _check_all_models_health(self):
        """Check health of all TTS models"""
        health_tasks = [
            self._check_model_health(TTSModel.HIGGS, self.higgs_url),
            self._check_model_health(TTSModel.CHATTERBOX, self.chatterbox_url),
            self._check_model_health(TTSModel.BARK, self.bark_url),
            self._check_model_health(TTSModel.KOKORO, self.kokoro_url),
        ]
        
        await asyncio.gather(*health_tasks, return_exceptions=True)
    
    async def _check_model_health(self, model: TTSModel, url: str):
        """Check health of a specific TTS model"""
        try:
            start_time = time.time()
            response = await self.client.get(f"{url}/health", timeout=5.0)
            latency_ms = int((time.time() - start_time) * 1000)
            
            is_healthy = response.status_code == 200
            
            self.model_health[model].is_healthy = is_healthy
            self.model_health[model].avg_latency_ms = latency_ms
            self.model_health[model].last_check = time.time()
            
            if is_healthy:
                logger.debug(f"✅ {model.value} is healthy ({latency_ms}ms)")
            else:
                logger.warning(f"⚠️ {model.value} is unhealthy")
                
        except Exception as e:
            logger.error(f"❌ Health check failed for {model.value}: {e}")
            self.model_health[model].is_healthy = False
            self.model_health[model].last_check = time.time()
    
    async def _health_monitor_loop(self):
        """Background task to monitor model health"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._check_all_models_health()
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    def _record_performance(self, model: TTSModel, latency_ms: int, success: bool):
        """Record performance metrics for a model"""
        self.performance_history[model].append(latency_ms)
        
        # Keep only last 100 measurements
        if len(self.performance_history[model]) > 100:
            self.performance_history[model] = self.performance_history[model][-100:]
        
        # Update average latency
        if self.performance_history[model]:
            self.model_health[model].avg_latency_ms = int(
                sum(self.performance_history[model]) / len(self.performance_history[model])
            )
    
    # Health check methods for individual services
    async def check_whisper_health(self) -> str:
        try:
            response = await self.client.get(f"{self.whisper_url}/health", timeout=5.0)
            return "healthy" if response.status_code == 200 else "unhealthy"
        except:
            return "unhealthy"
    
    async def check_higgs_health(self) -> str:
        return "healthy" if self.model_health[TTSModel.HIGGS].is_healthy else "unhealthy"
    
    async def check_chatterbox_health(self) -> str:
        return "healthy" if self.model_health[TTSModel.CHATTERBOX].is_healthy else "unhealthy"
    
    async def check_bark_health(self) -> str:
        return "healthy" if self.model_health[TTSModel.BARK].is_healthy else "unhealthy"
    
    async def check_kokoro_health(self) -> str:
        return "healthy" if self.model_health[TTSModel.KOKORO].is_healthy else "unhealthy"
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models"""
        stats = {}
        
        for model in TTSModel:
            health = self.model_health[model]
            history = self.performance_history[model]
            
            stats[model.value] = {
                "is_healthy": health.is_healthy,
                "avg_latency_ms": health.avg_latency_ms,
                "error_rate": health.error_rate,
                "quality_score": VOICE_MODELS[model.value]["quality_score"],
                "recent_latencies": history[-10:] if history else [],
                "total_requests": len(history)
            }
        
        return stats
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()
        logger.info("🔄 Smart Voice Engine cleanup completed")
