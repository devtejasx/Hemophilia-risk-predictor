"""
Clinical AI Chatbot using OpenAI GPT-4 API
With SQLite conversation history and comprehensive clinical response generation

Features:
- Patient context management
- Medical-style clinical responses
- Risk explanations, treatment suggestions, monitoring recommendations
- Conversation history tracking
- GPT-4 with fallback to GPT-3.5
- Medical disclaimer integration
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import openai
from pathlib import Path


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# OpenAI Configuration
openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
MODELS = ["gpt-4", "gpt-3.5-turbo"]  # Fallback order

# Medical Disclaimer
MEDICAL_DISCLAIMER = """
⚠️ MEDICAL DISCLAIMER:
This AI system provides clinical decision support based on patient data and medical literature.
It is NOT a substitute for professional medical judgment. All recommendations should be reviewed
and validated by qualified healthcare professionals before clinical implementation.
Use only in supervised clinical environments with proper oversight.
"""


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PatientContext:
    """Patient clinical information for context"""
    patient_id: str
    age: int
    gender: str
    diagnosis: str
    severity: str  # mild, moderate, severe
    treatment_history: List[str]
    current_medications: List[str]
    allergies: List[str]
    recent_labs: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_context_string(self) -> str:
        """Convert to formatted context for GPT"""
        context = f"""
PATIENT CONTEXT:
- ID: {self.patient_id}
- Age: {self.age} years
- Gender: {self.gender}
- Diagnosis: {self.diagnosis}
- Severity: {self.severity.upper()}
- Treatment History: {', '.join(self.treatment_history) if self.treatment_history else 'None'}
- Current Medications: {', '.join(self.current_medications) if self.current_medications else 'None'}
- Allergies: {', '.join(self.allergies) if self.allergies else 'No known allergies'}
"""
        if self.recent_labs:
            context += f"- Recent Labs: {json.dumps(self.recent_labs, indent=2)}\n"
        return context


@dataclass
class ConversationMessage:
    """Single conversation message"""
    message_id: int
    patient_id: str
    timestamp: datetime
    role: str  # 'user' or 'assistant'
    content: str
    message_type: str  # 'question', 'recommendation', 'explanation'
    context: Optional[Dict] = None


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

class ConversationDatabase:
    """SQLite database for storing conversation history"""
    
    def __init__(self, db_path: str = "clinical_conversations.db"):
        """
        Initialize conversation database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    patient_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_type TEXT,
                    context JSON,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)
            
            # Clinical notes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clinical_notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    note_type TEXT,
                    risk_score REAL,
                    recommendations TEXT,
                    treatment_suggestions TEXT,
                    monitoring_plan TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_id ON conversations(patient_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_id ON messages(conversation_id)")
            
            conn.commit()
    
    def start_conversation(self, patient_id: str) -> int:
        """
        Start new conversation
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Conversation ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (patient_id, status) VALUES (?, ?)",
                (patient_id, 'active')
            )
            conn.commit()
            return cursor.lastrowid
    
    def end_conversation(self, conversation_id: int):
        """End conversation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE conversations SET ended_at = CURRENT_TIMESTAMP, status = ? WHERE conversation_id = ?",
                ('ended', conversation_id)
            )
            conn.commit()
    
    def save_message(
        self,
        conversation_id: int,
        patient_id: str,
        role: str,
        content: str,
        message_type: str = None,
        context: Dict = None
    ) -> int:
        """
        Save message to database
        
        Args:
            conversation_id: Conversation ID
            patient_id: Patient ID
            role: 'user' or 'assistant'
            content: Message content
            message_type: Type of message
            context: Optional context data
            
        Returns:
            Message ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages 
                (conversation_id, patient_id, role, content, message_type, context)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                conversation_id,
                patient_id,
                role,
                content,
                message_type,
                json.dumps(context) if context else None
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_conversation_history(self, conversation_id: int) -> List[Dict]:
        """Get conversation history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
            """, (conversation_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_patient_conversations(self, patient_id: str) -> List[Dict]:
        """Get all conversations for patient"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE patient_id = ? 
                ORDER BY started_at DESC
            """, (patient_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def save_clinical_note(
        self,
        patient_id: str,
        conversation_id: int,
        note_type: str,
        risk_score: float = None,
        recommendations: str = None,
        treatment_suggestions: str = None,
        monitoring_plan: str = None
    ):
        """Save clinical note from conversation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO clinical_notes 
                (patient_id, conversation_id, note_type, risk_score, 
                 recommendations, treatment_suggestions, monitoring_plan)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id,
                conversation_id,
                note_type,
                risk_score,
                recommendations,
                treatment_suggestions,
                monitoring_plan
            ))
            conn.commit()


# ============================================================================
# OPENAI INTEGRATION
# ============================================================================

class ClinicalAIChatbot:
    """Clinical AI Chatbot with GPT-4 and fallback support"""
    
    def __init__(self, db_path: str = "clinical_conversations.db"):
        """
        Initialize chatbot
        
        Args:
            db_path: Database path for conversation history
        """
        self.db = ConversationDatabase(db_path)
        self.conversation_id = None
        self.patient_context: Optional[PatientContext] = None
        self.current_model = MODELS[0]  # Start with GPT-4
    
    def _call_openai(
        self,
        messages: List[Dict],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Tuple[str, str]:
        """
        Call OpenAI API with model fallback
        
        Args:
            messages: Conversation messages
            system_prompt: System message for context
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            
        Returns:
            Tuple of (response_text, model_used)
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        for model in MODELS:
            try:
                print(f"📡 Calling {model}...")
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                self.current_model = model
                return response.choices[0].message["content"], model
            
            except Exception as e:
                print(f"❌ {model} failed: {e}")
                if model == MODELS[-1]:  # Last model failed
                    raise Exception(f"All models failed: {e}")
                continue
    
    def start_session(self, patient_context: PatientContext) -> int:
        """
        Start chatbot session with patient context
        
        Args:
            patient_context: Patient clinical information
            
        Returns:
            Conversation ID
        """
        self.patient_context = patient_context
        self.conversation_id = self.db.start_conversation(patient_context.patient_id)
        
        print(f"\n{'='*60}")
        print(f"✅ New Clinical Session Started")
        print(f"Conversation ID: {self.conversation_id}")
        print(f"Patient: {patient_context.patient_id} (Age: {patient_context.age})")
        print(f"{'='*60}\n")
        
        return self.conversation_id
    
    def generate_risk_explanation(self, assessment_data: Dict) -> str:
        """
        Generate risk explanation with detailed analysis
        
        Args:
            assessment_data: Clinical assessment data with risk factors
                - key_risk_factors: List of risk factors
                - severity_score: 0-100 risk score
                - comorbidities: List of comorbid conditions
                - lab_abnormalities: Lab findings
                
        Returns:
            Risk explanation text
        """
        if not self.patient_context:
            raise ValueError("Patient context not set. Call start_session first.")
        
        system_prompt = """You are a clinical risk assessment expert. 
        Generate a detailed, evidence-based risk explanation based on patient data.
        Include:
        1. Summary of risk level
        2. Primary contributing factors
        3. Pathophysiological mechanisms
        4. Comparison to clinical benchmarks
        
        Use clear medical terminology with explanations for complex concepts."""
        
        patient_info = self.patient_context.to_context_string()
        
        user_message = f"""{patient_info}

RISK ASSESSMENT DATA:
- Key Risk Factors: {', '.join(assessment_data.get('key_risk_factors', []))}
- Severity Score: {assessment_data.get('severity_score', 0)}/100
- Comorbidities: {', '.join(assessment_data.get('comorbidities', [])) or 'None'}
- Lab Abnormalities: {', '.join(assessment_data.get('lab_abnormalities', [])) or 'None'}

Please generate a comprehensive risk explanation."""
        
        messages = [{"role": "user", "content": user_message}]
        
        risk_explanation, model_used = self._call_openai(
            messages,
            system_prompt,
            temperature=0.5,
            max_tokens=1500
        )
        
        # Save to database
        self.db.save_message(
            self.conversation_id,
            self.patient_context.patient_id,
            "assistant",
            risk_explanation,
            message_type="risk_explanation",
            context=assessment_data
        )
        
        return risk_explanation
    
    def get_clinical_recommendations(self, clinical_question: str) -> str:
        """
        Generate clinical recommendations and treatment suggestions
        
        Args:
            clinical_question: Question or topic for recommendations
                Example: "What are the best treatment options for this patient?"
                
        Returns:
            Clinical recommendations text
        """
        if not self.patient_context:
            raise ValueError("Patient context not set. Call start_session first.")
        
        system_prompt = """You are an experienced clinical specialist in hemophilia management.
        Provide evidence-based clinical recommendations based on:
        1. Current clinical guidelines
        2. Patient-specific factors
        3. Mechanism of action and efficacy
        4. Safety considerations and contraindications
        
        Format recommendations as:
        - RECOMMENDED APPROACH: [Best option]
        - ALTERNATIVE OPTIONS: [Other viable treatments]
        - RATIONALE: [Why this is recommended]
        - MONITORING: [What to monitor]
        - CONTRAINDICATIONS: [Avoid if...]"""
        
        patient_info = self.patient_context.to_context_string()
        
        user_message = f"""{patient_info}

CLINICAL QUESTION: {clinical_question}

Please provide detailed clinical recommendations."""
        
        # Save user question
        self.db.save_message(
            self.conversation_id,
            self.patient_context.patient_id,
            "user",
            clinical_question,
            message_type="clinical_question"
        )
        
        messages = [{"role": "user", "content": user_message}]
        
        recommendations, model_used = self._call_openai(
            messages,
            system_prompt,
            temperature=0.6,
            max_tokens=2000
        )
        
        # Save recommendations
        self.db.save_message(
            self.conversation_id,
            self.patient_context.patient_id,
            "assistant",
            recommendations,
            message_type="clinical_recommendation"
        )
        
        return recommendations
    
    def analyze_monitoring_data(self, monitoring_data: Dict) -> str:
        """
        Analyze monitoring data and provide clinical interpretation
        
        Args:
            monitoring_data: Patient monitoring data
                - lab_results: Dict of lab tests and values
                - vital_signs: Dict of vital signs
                - symptoms: List of reported symptoms
                - medication_adherence: Adherence percentage
                - adverse_events: List of AE if any
                
        Returns:
            Clinical analysis and interpretation
        """
        if not self.patient_context:
            raise ValueError("Patient context not set. Call start_session first.")
        
        system_prompt = """You are a clinical laboratory and data analyst.
        Analyze monitoring data and provide:
        1. INTERPRETATION: What the data shows
        2. ABNORMALITIES: Any concerning findings with clinical significance
        3. TRENDS: Analysis of changes over time if available
        4. CLINICAL IMPLICATIONS: What this means for patient care
        5. ACTIONS: Recommended clinical actions if needed
        6. FOLLOW-UP: Schedule for next monitoring"""
        
        patient_info = self.patient_context.to_context_string()
        
        monitoring_text = f"""
LAB RESULTS:
{json.dumps(monitoring_data.get('lab_results', {}), indent=2)}

VITAL SIGNS:
{json.dumps(monitoring_data.get('vital_signs', {}), indent=2)}

REPORTED SYMPTOMS:
{', '.join(monitoring_data.get('symptoms', [])) or 'No new symptoms'}

MEDICATION ADHERENCE: {monitoring_data.get('medication_adherence', 'Unknown')}%

ADVERSE EVENTS:
{', '.join(monitoring_data.get('adverse_events', [])) or 'None reported'}
"""
        
        user_message = f"""{patient_info}

MONITORING DATA:
{monitoring_text}

Please provide clinical interpretation and recommendations."""
        
        messages = [{"role": "user", "content": user_message}]
        
        analysis, model_used = self._call_openai(
            messages,
            system_prompt,
            temperature=0.5,
            max_tokens=1500
        )
        
        # Save analysis
        self.db.save_message(
            self.conversation_id,
            self.patient_context.patient_id,
            "assistant",
            analysis,
            message_type="monitoring_analysis",
            context=monitoring_data
        )
        
        return analysis
    
    def chat(self, user_message: str) -> str:
        """
        General chatbot conversation
        
        Args:
            user_message: User's question or statement
            
        Returns:
            Chatbot response
        """
        if not self.patient_context or not self.conversation_id:
            raise ValueError("Start session first with start_session()")
        
        # Save user message
        self.db.save_message(
            self.conversation_id,
            self.patient_context.patient_id,
            "user",
            user_message,
            message_type="question"
        )
        
        # Get conversation history
        history = self.db.get_conversation_history(self.conversation_id)
        messages = []
        
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        system_prompt = f"""You are a clinical decision support AI assistant expert in hemophilia care.
        You provide evidence-based clinical information and support clinical decision-making.
        
        {self.patient_context.to_context_string()}
        
        Key responsibilities:
        - Provide accurate clinical information based on current guidelines
        - Support clinical decision-making with evidence
        - Highlight important safety considerations
        - Always recommend professional medical review
        - Use clear, professional medical language
        
        {MEDICAL_DISCLAIMER}"""
        
        response, model_used = self._call_openai(
            messages,
            system_prompt,
            temperature=0.7,
            max_tokens=1500
        )
        
        # Save response
        self.db.save_message(
            self.conversation_id,
            self.patient_context.patient_id,
            "assistant",
            response,
            message_type="response"
        )
        
        return response
    
    def end_session(self) -> Dict:
        """
        End chatbot session and generate summary
        
        Returns:
            Session summary
        """
        if not self.conversation_id:
            raise ValueError("No active session")
        
        # Get conversation history
        history = self.db.get_conversation_history(self.conversation_id)
        
        self.db.end_conversation(self.conversation_id)
        
        summary = {
            "conversation_id": self.conversation_id,
            "patient_id": self.patient_context.patient_id,
            "total_messages": len(history),
            "messages_user": len([m for m in history if m["role"] == "user"]),
            "messages_assistant": len([m for m in history if m["role"] == "assistant"]),
            "timestamp_started": history[0]["timestamp"] if history else None,
            "model_used": self.current_model,
            "status": "ended"
        }
        
        print(f"\n{'='*60}")
        print(f"✅ Session Ended")
        print(f"Summary: {summary['total_messages']} messages exchanged")
        print(f"Model Used: {self.current_model}")
        print(f"{'='*60}\n")
        
        return summary


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_patient_context() -> PatientContext:
    """Create sample patient for testing"""
    return PatientContext(
        patient_id="PAT-2026-001",
        age=35,
        gender="M",
        diagnosis="Hemophilia A",
        severity="moderate",
        treatment_history=["Factor VIII replacement", "Prophylaxis therapy"],
        current_medications=["Factor VIII concentrate", "Aspirin"],
        allergies=["Penicillin"],
        recent_labs={
            "factor_viii_activity": "35%",
            "inhibitor_screen": "Negative",
            "platelet_count": "250K",
            "fibrinogen": "350 mg/dL"
        }
    )


def create_sample_assessment_data() -> Dict:
    """Create sample assessment data"""
    return {
        "key_risk_factors": [
            "Inhibitor development risk",
            "Joint damage progression",
            "Viral transmission risk"
        ],
        "severity_score": 65,
        "comorbidities": ["HIV positive"],
        "lab_abnormalities": ["Low factor VIII activity", "High inhibitor titer"]
    }


def create_sample_monitoring_data() -> Dict:
    """Create sample monitoring data"""
    return {
        "lab_results": {
            "factor_viii_activity": "40%",
            "inhibitor_level": "2 BU/mL",
            "hemoglobin": "14.2 g/dL",
            "platelet_count": "245K"
        },
        "vital_signs": {
            "blood_pressure": "120/80 mmHg",
            "heart_rate": "72 bpm",
            "temperature": "37.2°C"
        },
        "symptoms": ["Mild joint swelling", "Easy bruising"],
        "medication_adherence": 95,
        "adverse_events": []
    }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main function demonstrating chatbot capabilities"""
    
    print("\n" + "="*70)
    print("🏥 CLINICAL AI CHATBOT - HEMOPHILIA SUPPORT SYSTEM")
    print("="*70 + "\n")
    
    # Initialize chatbot
    chatbot = ClinicalAIChatbot(db_path="clinical_conversations.db")
    
    # Create sample patient
    patient = create_sample_patient_context()
    
    # Start session
    conversation_id = chatbot.start_session(patient)
    
    # ============================================================
    # DEMO 1: Generate Risk Explanation
    # ============================================================
    print("\n" + "-"*70)
    print("📋 DEMO 1: Risk Explanation Generation")
    print("-"*70 + "\n")
    
    assessment_data = create_sample_assessment_data()
    print("📊 Assessment Data:")
    print(json.dumps(assessment_data, indent=2))
    
    print("\n🔄 Generating risk explanation...")
    risk_explanation = chatbot.generate_risk_explanation(assessment_data)
    
    print("\n✅ Risk Explanation:")
    print(risk_explanation)
    
    # ============================================================
    # DEMO 2: Clinical Recommendations
    # ============================================================
    print("\n" + "-"*70)
    print("📋 DEMO 2: Clinical Recommendations")
    print("-"*70 + "\n")
    
    clinical_question = "What is the optimal prophylaxis strategy for this patient?"
    print(f"❓ Question: {clinical_question}\n")
    
    print("🔄 Generating recommendations...")
    recommendations = chatbot.get_clinical_recommendations(clinical_question)
    
    print("\n✅ Recommendations:")
    print(recommendations)
    
    # ============================================================
    # DEMO 3: Monitoring Data Analysis
    # ============================================================
    print("\n" + "-"*70)
    print("📋 DEMO 3: Monitoring Data Analysis")
    print("-"*70 + "\n")
    
    monitoring_data = create_sample_monitoring_data()
    print("📊 Monitoring Data:")
    print(json.dumps(monitoring_data, indent=2))
    
    print("\n🔄 Analyzing monitoring data...")
    analysis = chatbot.analyze_monitoring_data(monitoring_data)
    
    print("\n✅ Clinical Analysis:")
    print(analysis)
    
    # ============================================================
    # DEMO 4: General Chat Conversation
    # ============================================================
    print("\n" + "-"*70)
    print("📋 DEMO 4: General Clinical Chat")
    print("-"*70 + "\n")
    
    user_questions = [
        "What are the signs of an inhibitor development I should monitor for?",
        "How often should this patient have factor level testing?"
    ]
    
    for i, question in enumerate(user_questions, 1):
        print(f"\n👤 Patient Question {i}: {question}")
        print("\n🔄 Thinking...")
        
        response = chatbot.chat(question)
        
        print(f"\n🤖 Chatbot Response:")
        print(response)
    
    # ============================================================
    # End Session & Summary
    # ============================================================
    print("\n" + "-"*70)
    print("📋 SESSION SUMMARY")
    print("-"*70 + "\n")
    
    summary = chatbot.end_session()
    print(json.dumps(summary, indent=2))
    
    # ============================================================
    # Show Medical Disclaimer
    # ============================================================
    print("\n" + "="*70)
    print(MEDICAL_DISCLAIMER)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
