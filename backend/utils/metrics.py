"""
Metrics collection for Medical Voice Assistant
"""

import time
import logging
from typing import Dict, List, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ConversationMetrics:
    session_id: str
    timestamp: datetime
    user_message_length: int
    response_length: int
    latency_ms: int
    model_used: str
    confidence_score: float = 0.0
    medical_flags: List[str] = None


class MetricsCollector:
    """
    Collects and analyzes performance metrics for the voice assistant
    """
    
    def __init__(self):
        self.start_time = time.time()
        
        # Conversation metrics
        self.conversation_metrics: List[ConversationMetrics] = []
        self.session_metrics: Dict[str, List[ConversationMetrics]] = defaultdict(list)
        
        # Performance metrics
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.model_usage_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # System metrics
        self.total_conversations = 0
        self.total_sessions = 0
        self.active_sessions = set()
        
        # Quality metrics
        self.confidence_scores: deque = deque(maxlen=1000)
        self.medical_flags_frequency: Dict[str, int] = defaultdict(int)
        
    def record_conversation_turn(self, session_id: str, user_message_length: int,
                               response_length: int, latency_ms: int, model_used: str,
                               confidence_score: float = 0.0, medical_flags: List[str] = None):
        """Record a conversation turn"""
        
        try:
            # Create metrics entry
            metrics = ConversationMetrics(
                session_id=session_id,
                timestamp=datetime.now(),
                user_message_length=user_message_length,
                response_length=response_length,
                latency_ms=latency_ms,
                model_used=model_used,
                confidence_score=confidence_score,
                medical_flags=medical_flags or []
            )
            
            # Store metrics
            self.conversation_metrics.append(metrics)
            self.session_metrics[session_id].append(metrics)
            
            # Update counters
            self.total_conversations += 1
            self.active_sessions.add(session_id)
            
            # Update performance tracking
            self.latency_history[model_used].append(latency_ms)
            self.model_usage_counts[model_used] += 1
            self.confidence_scores.append(confidence_score)
            
            # Track medical flags
            for flag in (medical_flags or []):
                self.medical_flags_frequency[flag] += 1
            
            # Keep only recent conversation metrics (last 1000)
            if len(self.conversation_metrics) > 1000:
                self.conversation_metrics = self.conversation_metrics[-1000:]
            
            logger.debug(f"📊 Recorded metrics for session {session_id}: {latency_ms}ms, {model_used}")
            
        except Exception as e:
            logger.error(f"Failed to record conversation metrics: {e}")
    
    def record_error(self, error_type: str, component: str = "general"):
        """Record an error occurrence"""
        
        error_key = f"{component}:{error_type}"
        self.error_counts[error_key] += 1
        logger.debug(f"📊 Recorded error: {error_key}")
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time
    
    def get_average_latency(self, model: str = None) -> float:
        """Get average latency for a model or overall"""
        
        if model and model in self.latency_history:
            latencies = list(self.latency_history[model])
            return sum(latencies) / len(latencies) if latencies else 0.0
        
        # Overall average
        all_latencies = []
        for latencies in self.latency_history.values():
            all_latencies.extend(latencies)
        
        return sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for each model"""
        
        performance = {}
        
        for model, latencies in self.latency_history.items():
            if latencies:
                latency_list = list(latencies)
                performance[model] = {
                    "usage_count": self.model_usage_counts[model],
                    "avg_latency_ms": sum(latency_list) / len(latency_list),
                    "min_latency_ms": min(latency_list),
                    "max_latency_ms": max(latency_list),
                    "p95_latency_ms": self._percentile(latency_list, 95),
                    "recent_latencies": latency_list[-10:]  # Last 10 measurements
                }
        
        return performance
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        
        if not self.conversation_metrics:
            return {}
        
        recent_metrics = [m for m in self.conversation_metrics 
                         if (datetime.now() - m.timestamp).total_seconds() < 3600]  # Last hour
        
        return {
            "total_conversations": self.total_conversations,
            "active_sessions": len(self.active_sessions),
            "conversations_last_hour": len(recent_metrics),
            "avg_confidence_score": sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0,
            "medical_flags_frequency": dict(self.medical_flags_frequency),
            "avg_user_message_length": sum(m.user_message_length for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
            "avg_response_length": sum(m.response_length for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_rate": total_errors / max(self.total_conversations, 1),
            "errors_by_type": dict(self.error_counts)
        }
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        
        if session_id not in self.session_metrics:
            return {}
        
        session_data = self.session_metrics[session_id]
        
        if not session_data:
            return {}
        
        return {
            "session_id": session_id,
            "conversation_count": len(session_data),
            "session_duration": (session_data[-1].timestamp - session_data[0].timestamp).total_seconds(),
            "avg_latency_ms": sum(m.latency_ms for m in session_data) / len(session_data),
            "models_used": list(set(m.model_used for m in session_data)),
            "avg_confidence": sum(m.confidence_score for m in session_data) / len(session_data),
            "medical_flags": [flag for m in session_data for flag in (m.medical_flags or [])]
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a comprehensive report"""
        
        return {
            "system": {
                "uptime_seconds": self.get_uptime(),
                "uptime_formatted": self._format_duration(self.get_uptime()),
                "start_time": datetime.fromtimestamp(self.start_time).isoformat()
            },
            "performance": self.get_model_performance(),
            "conversations": self.get_conversation_stats(),
            "errors": self.get_error_stats(),
            "quality": {
                "avg_confidence_score": sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0,
                "confidence_distribution": self._get_confidence_distribution(),
                "medical_flags_frequency": dict(self.medical_flags_frequency)
            }
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for monitoring dashboard"""
        
        base_metrics = self.get_all_metrics()
        
        # Add real-time performance indicators
        base_metrics["realtime"] = {
            "current_active_sessions": len(self.active_sessions),
            "recent_avg_latency": self._get_recent_average_latency(),
            "recent_error_rate": self._get_recent_error_rate(),
            "model_health": self._get_model_health_indicators()
        }
        
        return base_metrics
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a dataset"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        
        duration = timedelta(seconds=int(seconds))
        
        if duration.days > 0:
            return f"{duration.days}d {duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m"
        elif duration.seconds >= 3600:
            return f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m"
        elif duration.seconds >= 60:
            return f"{duration.seconds // 60}m {duration.seconds % 60}s"
        else:
            return f"{duration.seconds}s"
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence scores"""
        
        if not self.confidence_scores:
            return {}
        
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for score in self.confidence_scores:
            if score >= 0.8:
                distribution["high"] += 1
            elif score >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def _get_recent_average_latency(self) -> float:
        """Get average latency for recent requests (last 10)"""
        
        recent_latencies = []
        
        for latencies in self.latency_history.values():
            recent_latencies.extend(list(latencies)[-10:])
        
        return sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0.0
    
    def _get_recent_error_rate(self) -> float:
        """Get error rate for recent activity"""
        
        recent_conversations = len([m for m in self.conversation_metrics 
                                  if (datetime.now() - m.timestamp).total_seconds() < 300])  # Last 5 minutes
        
        recent_errors = sum(1 for error_key, count in self.error_counts.items() 
                          if "recent" in error_key.lower())
        
        return recent_errors / max(recent_conversations, 1)
    
    def _get_model_health_indicators(self) -> Dict[str, str]:
        """Get health indicators for each model"""
        
        health = {}
        
        for model, latencies in self.latency_history.items():
            if not latencies:
                health[model] = "unknown"
                continue
            
            recent_latencies = list(latencies)[-5:]  # Last 5 requests
            avg_latency = sum(recent_latencies) / len(recent_latencies)
            
            if avg_latency < 200:
                health[model] = "excellent"
            elif avg_latency < 500:
                health[model] = "good"
            elif avg_latency < 1000:
                health[model] = "acceptable"
            else:
                health[model] = "poor"
        
        return health
    
    def reset_session_metrics(self, session_id: str):
        """Reset metrics for a specific session"""
        
        if session_id in self.session_metrics:
            del self.session_metrics[session_id]
        
        self.active_sessions.discard(session_id)
        logger.debug(f"📊 Reset metrics for session {session_id}")
    
    def cleanup_old_metrics(self, hours: int = 24):
        """Cleanup metrics older than specified hours"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter conversation metrics
            self.conversation_metrics = [
                m for m in self.conversation_metrics 
                if m.timestamp > cutoff_time
            ]
            
            # Filter session metrics
            for session_id in list(self.session_metrics.keys()):
                self.session_metrics[session_id] = [
                    m for m in self.session_metrics[session_id]
                    if m.timestamp > cutoff_time
                ]
                
                # Remove empty sessions
                if not self.session_metrics[session_id]:
                    del self.session_metrics[session_id]
            
            logger.info(f"🧹 Cleaned up metrics older than {hours} hours")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
