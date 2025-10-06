"""
Session Manager for Medical Voice Assistant
Handles user sessions, conversation state, and patient context
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis.asyncio as redis

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    session_id: str
    patient_id: Optional[str]
    doctor_id: Optional[str]
    conversation_type: str
    voice_sample: Optional[str]
    voice_id: str
    context: Dict[str, Any]
    created_at: datetime
    last_activity: datetime
    conversation_count: int = 0
    total_duration: float = 0.0


class SessionManager:
    """
    Manages user sessions and conversation state
    """
    
    def __init__(self, redis_url: str, max_sessions: int = 32):
        self.redis_url = redis_url
        self.max_sessions = max_sessions
        self.redis_client: Optional[redis.Redis] = None
        self.active_sessions: Dict[str, SessionData] = {}
        
    async def initialize(self):
        """Initialize Redis connection and session manager"""
        logger.info("🔄 Initializing Session Manager...")
        
        try:
            # Connect to Redis
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection successful")
            
            # Load existing sessions
            await self._load_existing_sessions()
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_expired_sessions())
            
            logger.info(f"✅ Session Manager initialized - {len(self.active_sessions)} active sessions")
            
        except Exception as e:
            logger.error(f"❌ Session Manager initialization failed: {e}")
            raise
    
    async def create_session(self, session_id: str, patient_id: Optional[str] = None,
                           doctor_id: Optional[str] = None, 
                           conversation_type: str = "general") -> SessionData:
        """Create a new session"""
        
        try:
            # Check if session already exists
            if session_id in self.active_sessions:
                logger.info(f"📋 Resuming existing session: {session_id}")
                session = self.active_sessions[session_id]
                session.last_activity = datetime.now()
                await self._save_session(session)
                return session
            
            # Check session limit
            if len(self.active_sessions) >= self.max_sessions:
                await self._cleanup_oldest_session()
            
            # Create new session
            session = SessionData(
                session_id=session_id,
                patient_id=patient_id,
                doctor_id=doctor_id,
                conversation_type=conversation_type,
                voice_sample=None,
                voice_id=self._get_default_voice_id(conversation_type),
                context={
                    "conversation_type": conversation_type,
                    "patient_preferences": {},
                    "medical_history": [],
                    "current_symptoms": [],
                    "conversation_flow": "greeting"
                },
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Store session
            self.active_sessions[session_id] = session
            await self._save_session(session)
            
            logger.info(f"✅ Created new session: {session_id} ({conversation_type})")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            raise
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data"""
        
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"Session not found for update: {session_id}")
                return False
            
            session = self.active_sessions[session_id]
            
            # Update session fields
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
                else:
                    # Add to context if not a direct field
                    session.context[key] = value
            
            # Update activity timestamp
            session.last_activity = datetime.now()
            session.conversation_count += 1
            
            # Save to Redis
            await self._save_session(session)
            
            logger.debug(f"📝 Updated session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data"""
        
        try:
            if session_id in self.active_sessions:
                return self.active_sessions[session_id]
            
            # Try to load from Redis
            session = await self._load_session(session_id)
            if session:
                self.active_sessions[session_id] = session
                return session
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def end_session(self, session_id: str) -> bool:
        """End a session"""
        
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Calculate total duration
            duration = (datetime.now() - session.created_at).total_seconds()
            session.total_duration = duration
            
            # Save final state to Redis with expiration
            await self._save_session(session, expire_hours=24)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"🔚 Ended session: {session_id} (duration: {duration:.1f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            return False
    
    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        
        try:
            if not self.redis_client:
                return []
            
            history_key = f"conversation_history:{session_id}"
            history_data = await self.redis_client.get(history_key)
            
            if history_data:
                return json.loads(history_data)
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get conversation history for {session_id}: {e}")
            return []
    
    async def add_conversation_turn(self, session_id: str, user_message: str,
                                 assistant_response: str, metadata: Dict = None):
        """Add a conversation turn to history"""
        
        try:
            if not self.redis_client:
                return
            
            history_key = f"conversation_history:{session_id}"
            
            # Get existing history
            existing_history = await self.get_conversation_history(session_id)
            
            # Add new turn
            turn = {
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response,
                "metadata": metadata or {}
            }
            
            existing_history.append(turn)
            
            # Keep only last 50 turns
            if len(existing_history) > 50:
                existing_history = existing_history[-50:]
            
            # Save back to Redis
            await self.redis_client.set(
                history_key,
                json.dumps(existing_history),
                ex=86400 * 7  # Expire in 7 days
            )
            
        except Exception as e:
            logger.error(f"Failed to add conversation turn for {session_id}: {e}")
    
    async def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.active_sessions)
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        
        try:
            total_sessions = len(self.active_sessions)
            
            # Count by conversation type
            type_counts = {}
            total_duration = 0.0
            
            for session in self.active_sessions.values():
                conv_type = session.conversation_type
                type_counts[conv_type] = type_counts.get(conv_type, 0) + 1
                total_duration += session.total_duration
            
            return {
                "total_active_sessions": total_sessions,
                "max_sessions": self.max_sessions,
                "sessions_by_type": type_counts,
                "average_duration": total_duration / max(total_sessions, 1),
                "total_duration": total_duration
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {}
    
    def _get_default_voice_id(self, conversation_type: str) -> str:
        """Get default voice ID based on conversation type"""
        
        voice_mapping = {
            "general": "medical_assistant_female",
            "emergency": "emergency_assistant",
            "consultation": "doctor_female",
            "follow_up": "nurse_female"
        }
        
        return voice_mapping.get(conversation_type, "medical_assistant_female")
    
    async def _save_session(self, session: SessionData, expire_hours: int = 2):
        """Save session to Redis"""
        
        try:
            if not self.redis_client:
                return
            
            session_key = f"session:{session.session_id}"
            session_data = {
                "session_id": session.session_id,
                "patient_id": session.patient_id,
                "doctor_id": session.doctor_id,
                "conversation_type": session.conversation_type,
                "voice_sample": session.voice_sample,
                "voice_id": session.voice_id,
                "context": session.context,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "conversation_count": session.conversation_count,
                "total_duration": session.total_duration
            }
            
            await self.redis_client.set(
                session_key,
                json.dumps(session_data),
                ex=expire_hours * 3600
            )
            
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
    
    async def _load_session(self, session_id: str) -> Optional[SessionData]:
        """Load session from Redis"""
        
        try:
            if not self.redis_client:
                return None
            
            session_key = f"session:{session_id}"
            session_data = await self.redis_client.get(session_key)
            
            if not session_data:
                return None
            
            data = json.loads(session_data)
            
            return SessionData(
                session_id=data["session_id"],
                patient_id=data.get("patient_id"),
                doctor_id=data.get("doctor_id"),
                conversation_type=data["conversation_type"],
                voice_sample=data.get("voice_sample"),
                voice_id=data["voice_id"],
                context=data["context"],
                created_at=datetime.fromisoformat(data["created_at"]),
                last_activity=datetime.fromisoformat(data["last_activity"]),
                conversation_count=data.get("conversation_count", 0),
                total_duration=data.get("total_duration", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    async def _load_existing_sessions(self):
        """Load existing sessions from Redis"""
        
        try:
            if not self.redis_client:
                return
            
            # Get all session keys
            session_keys = await self.redis_client.keys("session:*")
            
            for key in session_keys:
                session_id = key.split(":", 1)[1]
                session = await self._load_session(session_id)
                
                if session:
                    # Only load recent sessions (last 2 hours)
                    if (datetime.now() - session.last_activity).total_seconds() < 7200:
                        self.active_sessions[session_id] = session
            
            logger.info(f"📂 Loaded {len(self.active_sessions)} existing sessions")
            
        except Exception as e:
            logger.error(f"Failed to load existing sessions: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.active_sessions.items():
                    # Sessions expire after 2 hours of inactivity
                    if (current_time - session.last_activity).total_seconds() > 7200:
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    await self.end_session(session_id)
                
                if expired_sessions:
                    logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
                
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    async def _cleanup_oldest_session(self):
        """Remove the oldest session when limit is reached"""
        
        if not self.active_sessions:
            return
        
        # Find oldest session
        oldest_session_id = min(
            self.active_sessions.keys(),
            key=lambda sid: self.active_sessions[sid].last_activity
        )
        
        await self.end_session(oldest_session_id)
        logger.info(f"🧹 Removed oldest session due to limit: {oldest_session_id}")
    
    async def health_check(self) -> str:
        """Check health of session manager"""
        
        try:
            if self.redis_client:
                await self.redis_client.ping()
                return "healthy"
            else:
                return "unhealthy"
        except Exception as e:
            logger.error(f"Session manager health check failed: {e}")
            return "unhealthy"
    
    async def cleanup(self):
        """Cleanup resources"""
        
        try:
            # End all active sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.end_session(session_id)
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("🔄 Session Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
