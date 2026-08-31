"""
Chat Service
OpenAI GPT-4 integration for clinical responses
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
from config import settings
from exceptions import ChatException

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class ChatService:
    """Service for AI chatbot responses"""
    
    # System prompt for clinical context
    CLINICAL_SYSTEM_PROMPT = """
    You are an expert clinical decision support AI for hemophilia management.
    Your role is to provide evidence-based clinical insights, treatment recommendations, 
    and monitoring suggestions. Always include appropriate disclaimers that your responses 
    are educational and should be reviewed by healthcare professionals.
    
    Key responsibilities:
    - Provide clinical context and reasoning
    - Suggest evidence-based approaches
    - Highlight key considerations for patient safety
    - Recommend appropriate specialist consultation when needed
    - Always maintain patient confidentiality and HIPAA compliance
    
    Important: AI SUGGESTIONS ARE NOT MEDICAL ADVICE AND SHOULD NEVER REPLACE 
    PROFESSIONAL CLINICAL JUDGMENT.
    """
    
    def __init__(self):
        self.client = None
        self.model = settings.OPENAI_MODEL
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client"""
        if not settings.OPENAI_API_KEY:
            print("Warning: OpenAI API key not configured")
            return
        
        if OpenAI is None:
            print("Warning: OpenAI library not installed")
            return
        
        try:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI client: {e}")
    
    def get_clinical_response(
        self,
        user_message: str,
        patient_context: Optional[Dict[str, Any]] = None,
        mode: str = "general"
    ) -> str:
        """
        Get clinical response from GPT-4
        
        Args:
            user_message: User's question or request
            patient_context: Optional patient data for context
            mode: Response mode (general, diagnosis, treatment, monitoring)
            
        Returns:
            AI clinical response
        """
        if self.client is None or not settings.OPENAI_API_KEY:
            raise ChatException("OpenAI API not configured")
        
        try:
            # Build context string
            context_str = self._build_context(patient_context, mode)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": self.CLINICAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"{context_str}\n\nUser Question: {user_message}"}
            ]
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                timeout=60
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise ChatException(f"OpenAI API error: {str(e)}")
    
    def _build_context(self, patient_context: Optional[Dict], mode: str) -> str:
        """Build clinical context for the prompt"""
        context_parts = []
        
        if mode == "diagnosis":
            context_parts.append("Mode: Diagnostic Support - Provide differential diagnosis considerations")
        elif mode == "treatment":
            context_parts.append("Mode: Treatment Planning - Provide treatment optimization suggestions")
        elif mode == "monitoring":
            context_parts.append("Mode: Monitoring Guidance - Provide monitoring protocol suggestions")
        
        if patient_context:
            context_parts.append("\nPatient Context:")
            if "age" in patient_context:
                context_parts.append(f"- Age: {patient_context['age']}")
            if "severity" in patient_context:
                context_parts.append(f"- Severity: {patient_context['severity']}")
            if "mutation" in patient_context:
                context_parts.append(f"- Mutation: {patient_context['mutation']}")
            if "risk_score" in patient_context:
                context_parts.append(f"- Risk Score: {patient_context['risk_score']:.2f}")
        
        return "\n".join(context_parts)
    
    def is_available(self) -> bool:
        """Check if chat service is available"""
        return self.client is not None and bool(settings.OPENAI_API_KEY)


# Global chat service instance
chat_service = ChatService()
