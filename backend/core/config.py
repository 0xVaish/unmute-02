"""
Configuration for Ultra Human-Like Medical Voice Assistant
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings optimized for human-like voice"""
    
    # Basic Configuration
    APP_NAME: str = "Medical Voice Assistant - Ultra Human Voice"
    VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Voice Service URLs (Multi-TTS Architecture)
    WHISPER_URL: str = Field(default="http://localhost:8001", env="WHISPER_URL")
    HIGGS_URL: str = Field(default="http://localhost:8002", env="HIGGS_URL")
    CHATTERBOX_URL: str = Field(default="http://localhost:8003", env="CHATTERBOX_URL")
    KOKORO_URL: str = Field(default="http://localhost:8004", env="KOKORO_URL")
    BARK_URL: str = Field(default="http://localhost:8005", env="BARK_URL")
    
    # Smart Voice Selection Strategy
    PRIMARY_TTS: str = Field(default="higgs", env="PRIMARY_TTS")          # Best quality
    FALLBACK_TTS: str = Field(default="chatterbox", env="FALLBACK_TTS")   # Fast backup
    REALTIME_TTS: str = Field(default="kokoro", env="REALTIME_TTS")       # Ultra-fast
    NATURAL_TTS: str = Field(default="bark", env="NATURAL_TTS")           # Most natural/human-like
    AUTO_FALLBACK: bool = Field(default=True, env="AUTO_FALLBACK")
    MAX_LATENCY_MS: int = Field(default=200, env="MAX_LATENCY_MS")
    
    # Database Configuration
    DATABASE_URL: str = Field(env="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # LLM Configuration (Gemini Only)
    USE_GEMINI: bool = Field(default=True, env="USE_GEMINI")
    GOOGLE_API_KEY: str = Field(env="GOOGLE_API_KEY")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")
    
    # Medical Configuration
    MEDICAL_DOMAIN: str = Field(default="general_practice", env="MEDICAL_DOMAIN")
    ENABLE_EMERGENCY_DETECTION: bool = Field(default=True, env="ENABLE_EMERGENCY_DETECTION")
    ENABLE_MEDICATION_CHECKS: bool = Field(default=True, env="ENABLE_MEDICATION_CHECKS")
    ENABLE_SYMPTOM_TRACKING: bool = Field(default=True, env="ENABLE_SYMPTOM_TRACKING")
    
    # Voice Configuration
    VOICE_CLONING: bool = Field(default=True, env="VOICE_CLONING")
    EMOTION_CONTROL: bool = Field(default=True, env="EMOTION_CONTROL")
    SPEAKING_RATE: float = Field(default=1.0, env="SPEAKING_RATE")
    
    # Performance Settings
    MAX_CONCURRENT_SESSIONS: int = Field(default=32, env="MAX_CONCURRENT_SESSIONS")
    RESPONSE_TIMEOUT_MS: int = Field(default=5000, env="RESPONSE_TIMEOUT_MS")
    MAX_CONVERSATION_LENGTH: int = Field(default=3600, env="MAX_CONVERSATION_LENGTH")
    
    # Audio Processing
    SAMPLE_RATE: int = Field(default=16000, env="SAMPLE_RATE")
    CHUNK_SIZE: int = Field(default=1024, env="CHUNK_SIZE")
    AUDIO_FORMAT: str = Field(default="wav", env="AUDIO_FORMAT")
    MAX_AUDIO_LENGTH: int = Field(default=30, env="MAX_AUDIO_LENGTH")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="logs/medical_voice.log", env="LOG_FILE")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="ALLOWED_ORIGINS"
    )
    
    # Privacy & Security
    ENABLE_ENCRYPTION: bool = Field(default=True, env="ENABLE_ENCRYPTION")
    LOG_CONVERSATIONS: bool = Field(default=True, env="LOG_CONVERSATIONS")
    DATA_RETENTION_DAYS: int = Field(default=90, env="DATA_RETENTION_DAYS")
    HIPAA_MODE: bool = Field(default=True, env="HIPAA_MODE")
    SECRET_KEY: str = Field(default="change-in-production", env="SECRET_KEY")
    
    # Voice Activity Detection
    VAD_THRESHOLD: float = Field(default=0.5, env="VAD_THRESHOLD")
    SILENCE_DURATION: float = Field(default=1.0, env="SILENCE_DURATION")
    
    # Emergency Keywords
    EMERGENCY_KEYWORDS: List[str] = Field(
        default=[
            "emergency", "urgent", "chest pain", "difficulty breathing",
            "severe pain", "bleeding", "unconscious", "heart attack",
            "stroke", "allergic reaction", "overdose", "suicide"
        ],
        env="EMERGENCY_KEYWORDS"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Voice Model Configurations
VOICE_MODELS = {
    "higgs": {
        "name": "Higgs Audio V2",
        "model_id": "bosonai/higgs-audio-v2-generation-3B-base",
        "quality_score": 95,
        "avg_latency_ms": 450,
        "voice_cloning": True,
        "emotion_control": True,
        "best_for": ["consultations", "detailed_explanations", "empathetic_responses"],
        "gpu_memory_gb": 8,
        "batch_size": 1
    },
    "chatterbox": {
        "name": "Chatterbox Multilingual",
        "model_id": "ResembleAI/Chatterbox-Multilingual-TTS",
        "quality_score": 89,
        "avg_latency_ms": 280,
        "voice_cloning": True,
        "emotion_control": True,
        "best_for": ["quick_responses", "general_conversation", "multilingual"],
        "gpu_memory_gb": 4,
        "batch_size": 2
    },
    "bark": {
        "name": "Bark TTS",
        "model_id": "suno-ai/bark",
        "quality_score": 97,
        "avg_latency_ms": 600,
        "voice_cloning": True,
        "emotion_control": True,
        "best_for": ["most_natural", "human_like", "emotional_responses"],
        "gpu_memory_gb": 6,
        "batch_size": 1
    },
    "kokoro": {
        "name": "Kokoro v1.0",
        "model_id": "hexgrad/Kokoro-82M",
        "quality_score": 78,
        "avg_latency_ms": 120,
        "voice_cloning": False,
        "emotion_control": False,
        "best_for": ["real_time", "emergency_responses", "low_latency"],
        "gpu_memory_gb": 1,
        "batch_size": 4
    }
}

# Medical Voice Profiles
MEDICAL_VOICE_PROFILES = {
    "doctor_female": {
        "name": "Dr. Sarah",
        "description": "Professional, warm female doctor voice",
        "personality": "empathetic_professional",
        "use_cases": ["consultations", "diagnosis_discussion", "treatment_plans"]
    },
    "doctor_male": {
        "name": "Dr. Michael",
        "description": "Calm, authoritative male doctor voice",
        "personality": "confident_reassuring",
        "use_cases": ["emergency_guidance", "medical_procedures", "serious_discussions"]
    },
    "nurse_female": {
        "name": "Nurse Jenny",
        "description": "Gentle, caring female nurse voice",
        "personality": "nurturing_supportive",
        "use_cases": ["patient_care", "medication_reminders", "comfort"]
    },
    "assistant_neutral": {
        "name": "Medical Assistant",
        "description": "Professional, neutral assistant voice",
        "personality": "helpful_efficient",
        "use_cases": ["appointments", "information", "general_queries"]
    }
}

# Smart Selection Rules
VOICE_SELECTION_RULES = {
    "emergency": {
        "preferred_model": "kokoro",
        "max_latency_ms": 150,
        "priority": "speed"
    },
    "consultation": {
        "preferred_model": "higgs",
        "max_latency_ms": 500,
        "priority": "quality"
    },
    "quick_question": {
        "preferred_model": "chatterbox",
        "max_latency_ms": 300,
        "priority": "balanced"
    },
    "long_explanation": {
        "preferred_model": "higgs",
        "max_latency_ms": 600,
        "priority": "quality"
    },
    "real_time_chat": {
        "preferred_model": "kokoro",
        "max_latency_ms": 200,
        "priority": "speed"
    }
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "excellent": {"latency_ms": 150, "quality_score": 90},
    "good": {"latency_ms": 300, "quality_score": 80},
    "acceptable": {"latency_ms": 500, "quality_score": 70},
    "poor": {"latency_ms": 1000, "quality_score": 60}
}

# Create global settings instance
settings = Settings()
