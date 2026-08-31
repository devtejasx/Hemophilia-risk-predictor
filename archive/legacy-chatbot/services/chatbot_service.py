"""
Chatbot service for AI-powered conversations
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ChatbotService:
    """Service for handling chatbot interactions"""
    
    def __init__(self):
        self.conversation_history = []
        self.initialized = True
    
    def add_message(self, role: str, content: str) -> Dict[str, Any]:
        """Add message to conversation"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(message)
        return message
    
    def get_response(self, user_message: str) -> str:
        """Get bot response (placeholder - integrate with real LLM)"""
        try:
            # This is a placeholder - in production, integrate with OpenAI, Hugging Face, etc.
            responses = {
                "hello": "Hello! 👋 I'm the Hemophilia AI Assistant. How can I help you today?",
                "risk": "I can help assess patient risk scores. Please provide patient details for analysis.",
                "treatment": "For treatment recommendations, please consult with your medical team. I can provide data insights.",
                "prediction": "I can analyze clinical parameters and provide risk predictions. What patient would you like to analyze?",
                "help": "I can help with:\n- Patient risk assessment\n- Treatment insights\n- Data analysis\n- Clinical questions\n\nWhat would you like to know?",
                "default": "Thank you for your question. Could you provide more details or ask about a specific patient?",
            }
            
            user_lower = user_message.lower()
            
            for keyword, response in responses.items():
                if keyword in user_lower:
                    return response
            
            return responses["default"]
        
        except Exception as e:
            logger.error(f"Chatbot error: {e}")
            return "I encountered an error processing your request. Please try again."
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history
    
    def get_summary(self) -> str:
        """Get conversation summary"""
        if not self.conversation_history:
            return "No conversation history"
        
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        return f"Conversation with {len(user_messages)} user messages"


# Singleton instance
_chatbot_service = None


def get_chatbot_service() -> ChatbotService:
    """Get chatbot service instance"""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service
