"""
Chatbot service for clinical questions and patient guidance.
Consolidated chatbot logic for clinical decision support.
"""

from typing import Optional, List, Dict
import random


class ChatbotService:
    """AI chatbot service for clinical interaction."""
    
    def __init__(self):
        """Initialize chatbot service."""
        self.conversation_history = []
        self.context = {}
        
        # Knowledge base for clinical questions
        self.knowledge_base = self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self) -> Dict[str, List[str]]:
        """Initialize chatbot knowledge base with clinical information.
        
        Returns:
            Dictionary of keywords to responses
        """
        
        return {
            "clotting": [
                "Clotting factor levels are critical in hemophilia management. Normal levels are 50-150%. Low levels increase bleeding risk.",
                "Regular factor level monitoring helps adjust treatment dosage appropriately.",
                "Clotting factors include Factor VIII (Hemophilia A) and Factor IX (Hemophilia B).",
            ],
            "bleed": [
                "Bleeding episodes should be managed immediately with factor replacement therapy.",
                "Common bleeding sites include joints, muscles, and mucous membranes.",
                "Monitor for signs of internal bleeding: pain, bruising, swelling, or limited joint movement.",
                "Contact emergency services for severe bleeding or head/chest injuries.",
            ],
            "treatment": [
                "Treatment options include factor replacement, gene therapy (newer), desmopressin, and antifibrinolytics.",
                "Personalized treatment plans depend on severity, frequency of bleeds, and patient lifestyle.",
                "Regular prophylaxis (preventive treatment) is recommended for severe hemophilia.",
                "Physical therapy helps maintain joint function after bleeding episodes.",
            ],
            "prophylaxis": [
                "Prophylaxis is preventive factor replacement therapy given regularly.",
                "Typically administered 2-3 times per week for optimal bleeding prevention.",
                "Reduces frequency of spontaneous bleeds by 90%+ in many patients.",
                "Requires adherence but significantly improves quality of life.",
            ],
            "activity": [
                "Patients should avoid high-impact activities (contact sports, activities with fall risk).",
                "Swimming, walking, and cycling are generally safe with proper management.",
                "Always have factor product available when engaging in any activity.",
                "Regular exercise supports joint health and overall well-being.",
            ],
            "compliance": [
                "Treatment adherence is crucial for optimal outcomes and hemorrhage prevention.",
                "Set reminders for prophylaxis appointments and regular factor injections.",
                "Discuss barriers to compliance with your healthcare team.",
                "Home therapy programs improve convenience and compliance.",
            ],
            "pregnancy": [
                "Women with hemophilia can have successful pregnancies with proper planning.",
                "Close coordination with obstetrics and hematology teams is essential.",
                "Factor levels increase naturally during pregnancy, reducing bleeding risk.",
                "Carrier testing should be considered for family members.",
            ],
            "diagnosis": [
                "Hemophilia is diagnosed through coagulation studies and factor level testing.",
                "Common tests include aPTT, PT, bleeding time, and specific factor assays.",
                "Family history is important for diagnosis and genetic counseling.",
                "Genetic testing can identify mutations for accurate diagnosis.",
            ],
            "pain": [
                "Joint and muscle pain after bleeds can be managed with ice, compression, and elevation.",
                "Acetaminophen is usually safe; avoid NSAIDs due to bleeding risk.",
                "Physical therapy reduces chronic pain from repeated bleeds.",
                "Contact your physician if pain is severe or persistent.",
            ],
            "inhibitor": [
                "Inhibitors are antibodies that reduce factor effectiveness (occurs in some patients).",
                "Requires specialized treatment approaches including bypass agents.",
                "Regular inhibitor screening is recommended, especially after new factor exposure.",
                "Contact specialized hemophilia center immediately if inhibitor is detected.",
            ],
            "help": [
                "I can assist with questions about hemophilia management, treatment, and patient care.",
                "Ask about specific topics: clotting, bleeds, treatment, activity, compliance, and more.",
                "For emergencies, please contact your local emergency services immediately.",
                "Always consult your hematology team for specific medical decisions.",
            ],
            "emergency": [
                "For severe bleeding, contact emergency services immediately (911 in US).",
                "Administer factor replacement if available and trained.",
                "Immobilize and elevate the affected area.",
                "Do not delay seeking emergency care.",
            ]
        }
    
    def get_response(self, user_message: str, patient_context: Optional[Dict] = None) -> str:
        """Generate chatbot response to user message.
        
        Args:
            user_message: User input message
            patient_context: Optional patient data for context-aware responses
        
        Returns:
            Chatbot response
        """
        
        # Store in history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Clean and analyze message
        message_lower = user_message.lower()
        
        # Check for emergency keywords
        emergency_keywords = ["emergency", "severe", "911", "hospital", "ambulance"]
        if any(keyword in message_lower for keyword in emergency_keywords):
            response = "⚠️ EMERGENCY: Please call 911 or your local emergency services immediately. This is not a substitute for emergency care."
        
        # Check for specific topics
        else:
            response = self._find_matching_response(message_lower, patient_context)
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _find_matching_response(self, message: str, context: Optional[Dict]) -> str:
        """Find matching response from knowledge base.
        
        Args:
            message: User message (lowercase)
            context: Patient context
        
        Returns:
            Matching response or default response
        """
        
        # Check for keyword matches
        best_matches = []
        for keyword, responses in self.knowledge_base.items():
            if keyword in message:
                best_matches.extend(responses)
        
        # If matches found, return random one
        if best_matches:
            return random.choice(best_matches)
        
        # Default response if no matches
        return (
            "I'm here to help with hemophilia-related questions. "
            "You can ask about clotting, bleeding management, treatment options, activity levels, "
            "compliance, pregnancy, diagnosis, pain management, inhibitors, or request emergency help. "
            "For specific medical advice, always consult your hematology team."
        )
    
    def get_clinical_guidance(self, patient_data: Dict) -> str:
        """Provide clinical guidance based on patient data.
        
        Args:
            patient_data: Patient information
        
        Returns:
            Clinical guidance message
        """
        
        guidance = []
        
        # Clotting factor assessment
        cf = patient_data.get('clotting_factor', 50)
        if cf < 20:
            guidance.append("⚠️ Very low clotting factor - ensure prophylaxis compliance and factor availability.")
        elif cf < 50:
            guidance.append("⚠️ Low clotting factor - increased bleeding risk, monitor carefully.")
        
        # Compliance assessment
        compliance = patient_data.get('compliance', 0.8)
        if compliance < 0.6:
            guidance.append("⚠️ Low treatment compliance - discuss barriers with healthcare team.")
        
        # Activity assessment
        activity = patient_data.get('activity_level', 5)
        if activity > 8:
            guidance.append("📌 High activity level - ensure proper prophylaxis and avoid high-risk activities.")
        
        # Recent bleeds assessment
        bleeds = patient_data.get('bleeds', 0)
        if bleeds > 5:
            guidance.append("⚠️ Frequent recent bleeds - consider prophylaxis adjustment or factor inhibitor screening.")
        
        # Hospitalization
        if patient_data.get('hospitalization', False):
            guidance.append("📌 Recent hospitalization - ensure continuity of care and factor access.")
        
        if not guidance:
            guidance.append("✓ Patient is managing well within normal parameters.")
        
        return "\n".join(guidance)
    
    def get_recommendations(self, patient_data: Dict) -> List[str]:
        """Get clinical recommendations for patient.
        
        Args:
            patient_data: Patient information
        
        Returns:
            List of recommendations
        """
        
        recommendations = []
        
        # General recommendations
        recommendations.append("1. Maintain consistent factor replacement schedule as prescribed")
        recommendations.append("2. Track bleeding episodes and report patterns to your team")
        recommendations.append("3. Wear medical alert identification")
        
        # Activity-based
        if patient_data.get('activity_level', 5) > 6:
            recommendations.append("4. Ensure factor product is available before any activity")
            recommendations.append("5. Consider prophylactic factor dosing before exercise")
        
        # Compliance-based
        if patient_data.get('compliance', 0.8) < 0.7:
            recommendations.append("6. Set up medication reminders and discuss adherence challenges")
        
        # Safety
        recommendations.append("7. Keep emergency contact information accessible")
        recommendations.append("8. Schedule regular check-ups with your hematology team")
        
        return recommendations
    
    def get_conversation_summary(self) -> str:
        """Get summary of conversation so far.
        
        Returns:
            Conversation summary
        """
        
        if not self.conversation_history:
            return "No conversation yet."
        
        user_messages = sum(1 for msg in self.conversation_history if msg["role"] == "user")
        assistant_messages = sum(1 for msg in self.conversation_history if msg["role"] == "assistant")
        
        return f"Conversation with {user_messages} user message(s) and {assistant_messages} assistant response(s)"
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        """Get conversation history.
        
        Returns:
            List of messages in conversation
        """
        return self.conversation_history


# Global service instance
_chatbot_service = None


def get_chatbot_service() -> ChatbotService:
    """Get or create chatbot service instance."""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service


def get_response(message: str, patient_context: Optional[Dict] = None) -> str:
    """Convenience function to get chatbot response."""
    return get_chatbot_service().get_response(message, patient_context)


def get_clinical_guidance(patient_data: Dict) -> str:
    """Convenience function to get clinical guidance."""
    return get_chatbot_service().get_clinical_guidance(patient_data)
