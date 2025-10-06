"""
Medical Assistant with Gemini LLM Integration
Specialized for medical conversations with RAG support
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai
from pinecone import Pinecone
import httpx

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class MedicalResponse:
    response_text: str
    confidence_score: float
    medical_flags: List[str]
    conversation_state: str
    next_questions: List[str]
    emergency_level: int = 0  # 0=normal, 1=urgent, 2=emergency
    suggested_actions: List[str] = None


class MedicalAssistant:
    """
    Medical Assistant powered by Gemini with specialized medical knowledge
    """
    
    def __init__(self, use_gemini: bool = True, api_key: str = None):
        self.use_gemini = use_gemini
        self.api_key = api_key or settings.GOOGLE_API_KEY
        
        # Initialize Gemini
        if self.use_gemini:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Initialize Pinecone for RAG
        if settings.ENABLE_RAG:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index = self.pc.Index(settings.PINECONE_INDEX)
        
        # Medical knowledge and conversation state
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.patient_context: Dict[str, Dict] = {}
        
        # Medical prompts and templates
        self.system_prompt = self._build_medical_system_prompt()
        
    async def initialize(self):
        """Initialize the medical assistant"""
        logger.info("🏥 Initializing Medical Assistant with Gemini...")
        
        # Test Gemini connection
        if self.use_gemini:
            try:
                test_response = await self._generate_with_gemini("Hello, test connection.")
                logger.info("✅ Gemini connection successful")
            except Exception as e:
                logger.error(f"❌ Gemini connection failed: {e}")
                raise
        
        # Test Pinecone connection
        if settings.ENABLE_RAG:
            try:
                stats = self.index.describe_index_stats()
                logger.info(f"✅ Pinecone connected - {stats['total_vector_count']} vectors")
            except Exception as e:
                logger.warning(f"⚠️ Pinecone connection failed: {e}")
        
        logger.info("✅ Medical Assistant initialized")
    
    async def process_message(self, message: str, session_id: str, 
                            context: Dict = None) -> MedicalResponse:
        """
        Process a medical conversation message
        
        Args:
            message: User's message
            session_id: Session identifier
            context: Additional context (patient info, etc.)
        
        Returns:
            MedicalResponse with AI response and medical analysis
        """
        
        try:
            # Initialize conversation history if needed
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            # Add user message to history
            self.conversation_history[session_id].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Analyze message for medical flags
            medical_flags = await self._analyze_medical_flags(message)
            emergency_level = self._assess_emergency_level(message, medical_flags)
            
            # Get relevant medical knowledge if RAG is enabled
            relevant_context = ""
            if settings.ENABLE_RAG:
                relevant_context = await self._get_relevant_medical_knowledge(message)
            
            # Build conversation context
            conversation_context = self._build_conversation_context(
                session_id, message, context, relevant_context
            )
            
            # Generate response with Gemini
            response_text = await self._generate_medical_response(
                conversation_context, medical_flags, emergency_level
            )
            
            # Add assistant response to history
            self.conversation_history[session_id].append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
                "medical_flags": medical_flags,
                "emergency_level": emergency_level
            })
            
            # Generate follow-up questions
            next_questions = await self._generate_follow_up_questions(
                message, response_text, medical_flags
            )
            
            # Determine conversation state
            conversation_state = self._determine_conversation_state(
                medical_flags, emergency_level, len(self.conversation_history[session_id])
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                response_text, medical_flags, relevant_context
            )
            
            # Generate suggested actions
            suggested_actions = await self._generate_suggested_actions(
                medical_flags, emergency_level, response_text
            )
            
            return MedicalResponse(
                response_text=response_text,
                confidence_score=confidence_score,
                medical_flags=medical_flags,
                conversation_state=conversation_state,
                next_questions=next_questions,
                emergency_level=emergency_level,
                suggested_actions=suggested_actions
            )
            
        except Exception as e:
            logger.error(f"Error processing medical message: {e}")
            
            # Return safe fallback response
            return MedicalResponse(
                response_text="I apologize, but I'm having trouble processing your request right now. For any urgent medical concerns, please contact your healthcare provider or emergency services.",
                confidence_score=0.0,
                medical_flags=["system_error"],
                conversation_state="error",
                next_questions=["Would you like to try rephrasing your question?"],
                emergency_level=0
            )
    
    async def get_greeting(self, conversation_type: str = "general") -> str:
        """Get an appropriate medical greeting"""
        
        greetings = {
            "general": "Hello! I'm your medical voice assistant. I'm here to help with your health questions and concerns. How can I assist you today?",
            "emergency": "Hello, I understand you may have an urgent medical concern. Please describe what's happening, and I'll do my best to help guide you.",
            "follow_up": "Hello again! How are you feeling since our last conversation? Any updates on your symptoms or concerns?",
            "consultation": "Hello! I'm here to help with your medical consultation. Please feel free to describe your symptoms or ask any health-related questions."
        }
        
        return greetings.get(conversation_type, greetings["general"])
    
    def _build_medical_system_prompt(self) -> str:
        """Build the system prompt for medical conversations"""
        
        return f"""You are a professional medical voice assistant designed to help patients with health-related questions and concerns. 

IMPORTANT GUIDELINES:
1. You are NOT a replacement for professional medical care
2. Always recommend consulting healthcare providers for serious concerns
3. Be empathetic, clear, and professional in your responses
4. Ask relevant follow-up questions to better understand symptoms
5. Provide helpful general health information when appropriate
6. Recognize emergency situations and recommend immediate care

MEDICAL SPECIALIZATION: {settings.MEDICAL_DOMAIN}

EMERGENCY DETECTION: When you detect potential emergency situations (chest pain, difficulty breathing, severe injuries, etc.), immediately recommend seeking emergency medical care.

CONVERSATION STYLE:
- Warm and professional tone
- Use simple, clear language
- Show empathy and understanding
- Ask one question at a time
- Provide actionable guidance when possible

LIMITATIONS:
- Cannot diagnose medical conditions
- Cannot prescribe medications
- Cannot replace professional medical advice
- Should always encourage professional consultation for serious concerns

Remember: You're having a voice conversation, so keep responses conversational and not too long (under 100 words typically).
"""
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    async def _analyze_medical_flags(self, message: str) -> List[str]:
        """Analyze message for medical flags and concerns"""
        
        flags = []
        message_lower = message.lower()
        
        # Emergency keywords
        emergency_keywords = [
            "chest pain", "heart attack", "can't breathe", "difficulty breathing",
            "severe pain", "bleeding heavily", "unconscious", "stroke",
            "allergic reaction", "overdose", "suicide", "emergency"
        ]
        
        for keyword in emergency_keywords:
            if keyword in message_lower:
                flags.append("emergency")
                break
        
        # Symptom keywords
        symptom_keywords = {
            "pain": ["pain", "hurt", "ache", "sore"],
            "fever": ["fever", "hot", "temperature", "chills"],
            "respiratory": ["cough", "breathing", "shortness of breath", "wheezing"],
            "digestive": ["nausea", "vomiting", "diarrhea", "stomach"],
            "neurological": ["headache", "dizzy", "confusion", "memory"],
            "mental_health": ["anxious", "depressed", "stress", "worried"]
        }
        
        for flag_type, keywords in symptom_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                flags.append(flag_type)
        
        # Medication keywords
        medication_keywords = ["medication", "medicine", "prescription", "pill", "drug"]
        if any(keyword in message_lower for keyword in medication_keywords):
            flags.append("medication")
        
        return flags
    
    def _assess_emergency_level(self, message: str, medical_flags: List[str]) -> int:
        """Assess emergency level (0=normal, 1=urgent, 2=emergency)"""
        
        if "emergency" in medical_flags:
            return 2
        
        urgent_indicators = [
            "severe", "intense", "unbearable", "getting worse",
            "can't", "unable", "difficulty", "trouble"
        ]
        
        message_lower = message.lower()
        if any(indicator in message_lower for indicator in urgent_indicators):
            return 1
        
        return 0
    
    async def _get_relevant_medical_knowledge(self, query: str) -> str:
        """Get relevant medical knowledge using RAG"""
        
        try:
            # Create embedding for the query (you'd need to implement this)
            # For now, return empty string
            # TODO: Implement proper embedding and vector search
            return ""
            
        except Exception as e:
            logger.error(f"RAG knowledge retrieval error: {e}")
            return ""
    
    def _build_conversation_context(self, session_id: str, current_message: str,
                                  context: Dict, relevant_knowledge: str) -> str:
        """Build conversation context for Gemini"""
        
        # Get recent conversation history
        history = self.conversation_history.get(session_id, [])
        recent_history = history[-6:]  # Last 3 exchanges
        
        context_parts = [self.system_prompt]
        
        # Add relevant medical knowledge if available
        if relevant_knowledge:
            context_parts.append(f"RELEVANT MEDICAL KNOWLEDGE:\n{relevant_knowledge}")
        
        # Add conversation history
        if recent_history:
            context_parts.append("CONVERSATION HISTORY:")
            for msg in recent_history:
                role = "Patient" if msg["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {msg['content']}")
        
        # Add current message
        context_parts.append(f"CURRENT PATIENT MESSAGE: {current_message}")
        
        # Add instructions
        context_parts.append("""
INSTRUCTIONS:
- Respond as a professional medical assistant
- Keep response under 100 words for voice conversation
- Be empathetic and helpful
- Ask relevant follow-up questions
- Recommend professional care when appropriate
- If emergency indicators present, prioritize safety guidance

RESPONSE:""")
        
        return "\n\n".join(context_parts)
    
    async def _generate_medical_response(self, context: str, medical_flags: List[str],
                                       emergency_level: int) -> str:
        """Generate medical response using Gemini"""
        
        # Add emergency handling if needed
        if emergency_level >= 2:
            emergency_context = context + "\n\nIMPORTANT: This appears to be a potential emergency. Prioritize safety and recommend immediate medical attention."
            response = await self._generate_with_gemini(emergency_context)
        else:
            response = await self._generate_with_gemini(context)
        
        return response.strip()
    
    async def _generate_follow_up_questions(self, user_message: str, 
                                          assistant_response: str,
                                          medical_flags: List[str]) -> List[str]:
        """Generate relevant follow-up questions"""
        
        questions = []
        
        # Based on medical flags
        if "pain" in medical_flags:
            questions.extend([
                "Can you describe the pain - is it sharp, dull, or throbbing?",
                "On a scale of 1-10, how would you rate the pain?",
                "When did the pain start?"
            ])
        
        if "fever" in medical_flags:
            questions.extend([
                "Have you taken your temperature?",
                "Are you experiencing any other symptoms along with the fever?"
            ])
        
        if "respiratory" in medical_flags:
            questions.extend([
                "Are you having any difficulty breathing?",
                "Is the cough dry or producing mucus?"
            ])
        
        # General follow-up questions
        general_questions = [
            "Is there anything else you'd like to discuss about your health?",
            "Do you have any other symptoms I should know about?",
            "How long have you been experiencing this?"
        ]
        
        # Return up to 2 most relevant questions
        all_questions = questions + general_questions
        return all_questions[:2]
    
    def _determine_conversation_state(self, medical_flags: List[str], 
                                    emergency_level: int, turn_count: int) -> str:
        """Determine current conversation state"""
        
        if emergency_level >= 2:
            return "emergency"
        elif emergency_level == 1:
            return "urgent"
        elif turn_count <= 2:
            return "initial_assessment"
        elif any(flag in medical_flags for flag in ["pain", "fever", "respiratory"]):
            return "symptom_discussion"
        else:
            return "general_consultation"
    
    def _calculate_confidence_score(self, response: str, medical_flags: List[str],
                                  relevant_knowledge: str) -> float:
        """Calculate confidence score for the response"""
        
        base_score = 0.7
        
        # Increase confidence if we have relevant knowledge
        if relevant_knowledge:
            base_score += 0.1
        
        # Increase confidence for general health topics
        if not medical_flags or len(medical_flags) == 1:
            base_score += 0.1
        
        # Decrease confidence for complex medical situations
        if len(medical_flags) > 2:
            base_score -= 0.1
        
        # Decrease confidence for emergency situations
        if "emergency" in medical_flags:
            base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))
    
    async def _generate_suggested_actions(self, medical_flags: List[str],
                                        emergency_level: int, response: str) -> List[str]:
        """Generate suggested actions based on the conversation"""
        
        actions = []
        
        if emergency_level >= 2:
            actions.extend([
                "Call emergency services (911) immediately",
                "Go to the nearest emergency room",
                "Contact your doctor right away"
            ])
        elif emergency_level == 1:
            actions.extend([
                "Contact your healthcare provider today",
                "Monitor symptoms closely",
                "Seek medical attention if symptoms worsen"
            ])
        else:
            actions.extend([
                "Schedule an appointment with your doctor if symptoms persist",
                "Keep track of your symptoms",
                "Follow general health guidelines"
            ])
        
        return actions[:3]  # Return top 3 actions
    
    async def health_check(self) -> str:
        """Check health of the medical assistant"""
        try:
            if self.use_gemini:
                # Test Gemini connection
                await self._generate_with_gemini("Test")
            return "healthy"
        except Exception as e:
            logger.error(f"Medical assistant health check failed: {e}")
            return "unhealthy"
    
    async def cleanup(self):
        """Cleanup resources"""
        # Clear conversation history to free memory
        self.conversation_history.clear()
        self.patient_context.clear()
        logger.info("🔄 Medical Assistant cleanup completed")
