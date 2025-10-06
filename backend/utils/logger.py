"""
Logging utilities for Medical Voice Assistant
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

from core.config import settings


def setup_logger(name: str) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    logger = logging.getLogger(name)
    
    # Don't add handlers if already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if settings.LOG_FILE:
        # Create log directory if it doesn't exist
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_conversation_turn(session_id: str, user_message: str, 
                         assistant_response: str, model_used: str, 
                         latency_ms: int):
    """Log conversation turn for analysis"""
    
    if not settings.LOG_CONVERSATIONS:
        return
    
    conversation_logger = logging.getLogger("conversation")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_message": user_message[:100] + "..." if len(user_message) > 100 else user_message,
        "response_length": len(assistant_response),
        "model_used": model_used,
        "latency_ms": latency_ms
    }
    
    conversation_logger.info(f"CONVERSATION: {log_entry}")


def log_performance_metric(metric_name: str, value: float, session_id: str = None):
    """Log performance metrics"""
    
    metrics_logger = logging.getLogger("metrics")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "metric": metric_name,
        "value": value,
        "session_id": session_id
    }
    
    metrics_logger.info(f"METRIC: {log_entry}")
