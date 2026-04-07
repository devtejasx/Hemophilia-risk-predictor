"""
Advanced Clinical AI Assistant with Structured Modes
======================================================

Provides context-aware clinical decision support with specialized modes:
- Diagnosis Support: Help interpreting symptoms and findings
- Treatment Recommendation: Suggest management strategies
- Risk Explanation: Explain risk factors and predictions
- Monitoring Analysis: Guide monitoring protocols

Features:
- Patient data context integration
- Structured prompt templates
- Safety disclaimers
- Medical terminology explanations
- Evidence-based recommendations
"""

import os
from openai import OpenAI
from typing import Dict, List, Optional, Tuple

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


class ClinicalAssistantMode:
    """Enum-like class for assistant modes"""
    DIAGNOSIS_SUPPORT = "diagnosis_support"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    RISK_EXPLANATION = "risk_explanation"
    MONITORING_ANALYSIS = "monitoring_analysis"


class StructuredPromptTemplates:
    """Prompt templates for each clinical mode"""
    
    @staticmethod
    def diagnosis_support(patient_context: Dict, question: str) -> str:
        """Template for diagnosis support mode"""
        return f"""You are an AI clinical decision support assistant for hemophilia care.

PATIENT CONTEXT:
- Name: {patient_context.get('Name', 'Patient')}
- Age: {patient_context.get('Age', 'Unknown')} years
- Hemophilia Severity: {patient_context.get('Severity', 'Unknown')}
- Mutation Type: {patient_context.get('Mutation', 'Unknown')}
- Current Treatment: {patient_context.get('Dose', 'N/A')} units, {patient_context.get('Exposure', 'N/A')} days exposure
- Family History: {patient_context.get('Family_History', 'Unknown')}
- Previous Inhibitor: {patient_context.get('Previous_Inhibitor', 'Unknown')}

MODE: DIAGNOSIS SUPPORT
You are helping interpret clinical findings and symptoms in the context of hemophilia care.

GUIDELINES:
1. Ask clarifying questions about symptoms, timing, and associated findings
2. Consider hemophilia-specific differential diagnoses
3. Explain findings in patient-friendly terms with medical definitions
4. Recommend when specialist consultation is needed
5. Provide evidence-based clinical reasoning

⚠️ CRITICAL DISCLAIMER:
"AI-generated suggestions are for educational purposes only and should not replace professional medical judgment. Always consult with qualified hematologists for clinical decisions."

PATIENT QUESTION: {question}

Provide a thorough, clinically appropriate response:"""

    @staticmethod
    def treatment_recommendation(patient_context: Dict, question: str) -> str:
        """Template for treatment recommendation mode"""
        return f"""You are an AI clinical decision support assistant for hemophilia management.

PATIENT CONTEXT:
- Name: {patient_context.get('Name', 'Patient')}
- Age: {patient_context.get('Age', 'Unknown')} years  
- Hemophilia Severity: {patient_context.get('Severity', 'Unknown')}
- Mutation Type: {patient_context.get('Mutation', 'Unknown')}
- Current Dose: {patient_context.get('Dose', 'N/A')} units
- Exposure Days: {patient_context.get('Exposure', 'N/A')}
- Treatment Adherence: {patient_context.get('Treatment_Adherence', 'Unknown')}%
- Product Type: {patient_context.get('Product_Type', 'Unknown')}
- Joint Damage Score: {patient_context.get('Joint_Damage', 'Unknown')}/124
- Inhibitor Risk: {patient_context.get('Risk', 'Unknown')}

MODE: TREATMENT RECOMMENDATION
You are helping evaluate and optimize hemophilia treatment strategies.

GUIDELINES:
1. Assess current treatment appropriateness for severity level
2. Consider inhibitor risk and previous episodes
3. Address treatment adherence barriers
4. Evaluate prophylaxis vs on-demand strategies
5. Discuss management of joint complications
6. Consider lifestyle and physical activity
7. Review product selection and alternatives

⚠️ CRITICAL DISCLAIMER:
"AI-generated suggestions are for educational and discussion purposes only. Treatment decisions must be made by qualified hematologists in consultation with the patient."

PATIENT QUESTION: {question}

Provide specific, evidence-based treatment guidance:"""

    @staticmethod
    def risk_explanation(patient_context: Dict, question: str) -> str:
        """Template for risk explanation mode"""
        return f"""You are an AI clinical decision support assistant specializing in hemophilia risk assessment.

PATIENT CONTEXT:
- Name: {patient_context.get('Name', 'Patient')}
- Age: {patient_context.get('Age', 'Unknown')} years
- Severity: {patient_context.get('Severity', 'Unknown')}
- Mutation: {patient_context.get('Mutation', 'Unknown')}
- Inhibitor Risk Score: {patient_context.get('Risk', 'Unknown')}
- Family History of Inhibitors: {patient_context.get('Family_History', 'Unknown')}
- Previous Inhibitor Episode: {patient_context.get('Previous_Inhibitor', 'Unknown')}
- Dose: {patient_context.get('Dose', 'N/A')} units
- Exposure: {patient_context.get('Exposure', 'N/A')} days
- Joint Damage: {patient_context.get('Joint_Damage', 'Unknown')}/124
- Bleeding Episodes: {patient_context.get('Bleeding_Episodes', 'Unknown')}/year

MODE: RISK EXPLANATION
You are explaining clinical risks and protective factors in hemophilia care.

GUIDELINES:
1. Explain what the risk score means in clinical terms
2. Identify top 3-5 risk factors contributing to elevated risk
3. Explain protective factors and adherence benefits
4. Provide evidence-based risk stratification information
5. Discuss prevention strategies for identified risks
6. Explain natural history and expected outcomes
7. Discuss genetic counseling for family members

⚠️ CRITICAL DISCLAIMER:
"This AI risk assessment is for educational discussion only. Medical decisions should be guided by qualified hematologists who can integrate clinical judgment, current guidelines, and individual circumstances."

PATIENT QUESTION: {question}

Explain risks clearly and comprehensively:"""

    @staticmethod
    def monitoring_analysis(patient_context: Dict, question: str) -> str:
        """Template for monitoring analysis mode"""
        return f"""You are an AI clinical decision support assistant for hemophilia monitoring and surveillance.

PATIENT CONTEXT:
- Name: {patient_context.get('Name', 'Patient')}
- Age: {patient_context.get('Age', 'Unknown')} years
- Severity: {patient_context.get('Severity', 'Unknown')}
- Risk Level: {patient_context.get('Risk', 'Unknown')}
- Factor Level: {patient_context.get('Factor_Level', 'Unknown')}%
- Treatment Product: {patient_context.get('Product_Type', 'Unknown')}
- Dose: {patient_context.get('Dose', 'N/A')} units
- Joint Damage Score: {patient_context.get('Joint_Damage', 'Unknown')}/124
- Annual Bleeds: {patient_context.get('Bleeding_Episodes', 'Unknown')}

MODE: MONITORING ANALYSIS
You are guiding optimal monitoring and surveillance strategies.

GUIDELINES:
1. Recommend appropriate monitoring frequency based on risk level
2. Explain key lab tests: factor levels, inhibitor screening, HJHS
3. Discuss physical exam assessments and joint monitoring
4. Address when increased monitoring is indicated
5. Explain normal ranges and concerning findings
6. Discuss home monitoring and self-advocacy
7. Provide guidance on specialist referral triggers

⚠️ CRITICAL DISCLAIMER:
"These AI-generated monitoring recommendations are educational and should be reviewed by your hematology team. Monitoring protocols should follow institutional guidelines and individual clinical circumstances."

PATIENT QUESTION: {question}

Provide comprehensive monitoring guidance:"""


class StructuredClinicalAssistant:
    """Advanced clinical assistant with structured modes"""
    
    def __init__(self):
        """Initialize the assistant"""
        self.modes = ClinicalAssistantMode
        self.templates = StructuredPromptTemplates()
    
    def get_mode_description(self, mode: str) -> str:
        """Get human-readable description of a mode"""
        descriptions = {
            self.modes.DIAGNOSIS_SUPPORT: "Help interpreting symptoms and clinical findings",
            self.modes.TREATMENT_RECOMMENDATION: "Optimize treatment strategies and management",
            self.modes.RISK_EXPLANATION: "Understand risk factors and predictions",
            self.modes.MONITORING_ANALYSIS: "Guide monitoring protocols and surveillance"
        }
        return descriptions.get(mode, "Unknown mode")
    
    def get_mode_icon(self, mode: str) -> str:
        """Get emoji icon for a mode"""
        icons = {
            self.modes.DIAGNOSIS_SUPPORT: "🔍",
            self.modes.TREATMENT_RECOMMENDATION: "💊",
            self.modes.RISK_EXPLANATION: "⚠️",
            self.modes.MONITORING_ANALYSIS: "📊"
        }
        return icons.get(mode, "🤖")
    
    def build_prompt(self, mode: str, patient_context: Dict, question: str) -> str:
        """Build appropriate prompt based on mode"""
        if mode == self.modes.DIAGNOSIS_SUPPORT:
            return self.templates.diagnosis_support(patient_context, question)
        elif mode == self.modes.TREATMENT_RECOMMENDATION:
            return self.templates.treatment_recommendation(patient_context, question)
        elif mode == self.modes.RISK_EXPLANATION:
            return self.templates.risk_explanation(patient_context, question)
        elif mode == self.modes.MONITORING_ANALYSIS:
            return self.templates.monitoring_analysis(patient_context, question)
        else:
            # Default mode
            return f"""You are an AI clinical decision support assistant for hemophilia care.

PATIENT CONTEXT:
{self._format_patient_context(patient_context)}

QUESTION: {question}

Provide helpful, evidence-based clinical information:"""
    
    def _format_patient_context(self, patient_context: Dict) -> str:
        """Format patient context for display"""
        if not patient_context:
            return "No patient data available. Providing general hemophilia information."
        
        context_lines = []
        for key, value in patient_context.items():
            if value:
                key_formatted = key.replace('_', ' ').title()
                context_lines.append(f"- {key_formatted}: {value}")
        
        return "\n".join(context_lines) if context_lines else "Limited patient data available"
    
    def generate_response(
        self,
        mode: str,
        question: str,
        patient_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """
        Generate response in specified mode
        
        Args:
            mode: Clinical assistant mode
            question: User's question
            patient_context: Patient data dictionary
            conversation_history: Previous messages for context
            
        Returns:
            Tuple of (response, mode_used)
        """
        try:
            # Build the system prompt based on mode
            system_prompt = self.build_prompt(
                mode,
                patient_context or {},
                question
            )
            
            # Prepare messages
            messages = []
            
            # Add conversation history if available
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Add current question
            messages.append({
                "role": "user",
                "content": question
            })
            
            # Call GPT API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages
                ],
                temperature=0.7,
                max_tokens=1500,
                top_p=0.95
            )
            
            return response.choices[0].message.content, mode
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "invalid_api_key" in error_msg:
                return (
                    "⚠️ **API Key Error**: OpenAI API key not configured. "
                    "Please set OPENAI_API_KEY in your .env file.",
                    mode
                )
            elif "429" in error_msg or "rate_limit" in error_msg:
                return (
                    "⚠️ **Rate Limited**: Too many requests. Please wait a moment and try again.",
                    mode
                )
            else:
                return (
                    f"⚠️ **Error**: {error_msg[:100]}. Please try again.",
                    mode
                )
    
    def get_medical_definitions(self, terms: List[str]) -> Dict[str, str]:
        """Get patient-friendly definitions of medical terms"""
        definitions = {
            "inhibitor": "Antibodies that neutralize clotting factor, reducing treatment effectiveness",
            "mutation": "A change in the DNA that affects factor production",
            "prophylaxis": "Regular preventive factor treatments to prevent bleeding",
            "bolus": "A single dose of medication given at one time",
            "hemostasis": "The body's ability to stop bleeding",
            "thrombosis": "Formation of unwanted blood clots",
            "arthropathy": "Joint disease or damage from repeated bleeding",
            "synovitis": "Inflammation of the joint membrane",
            "inhibitor titer": "Measurement of inhibitor antibody strength",
            "bethesda assay": "Test that measures inhibitor presence and level",
            "factor level": "Percentage of clotting factor in the blood",
            "half-life": "Time it takes for factor amount to decrease by half",
            "pharmacokinetics": "How the body absorbs and eliminates medication",
            "breakthrough bleed": "Spontaneous bleeding despite prophylactic treatment",
            "on-demand therapy": "Treatment given only when bleeding occurs",
            "joint health": "Overall condition and function of joints",
            "target joint": "Specific joint prone to repeated bleeds",
        }
        
        result = {}
        for term in terms:
            term_lower = term.lower()
            result[term] = definitions.get(
                term_lower,
                f"Medical term not in quick reference. Please ask for detailed explanation."
            )
        
        return result


# Initialize assistant
structured_assistant = StructuredClinicalAssistant()


def get_clinical_response(
    mode: str,
    question: str,
    patient_data: Optional[Dict] = None,
    conversation_history: Optional[List[Dict]] = None
) -> Tuple[str, str]:
    """
    Get clinical response in specified mode
    
    Args:
        mode: Clinical mode (diagnosis, treatment, risk, monitoring)
        question: User's clinical question
        patient_data: Patient context dictionary
        conversation_history: Previous messages
        
    Returns:
        Tuple of (response, mode_used)
    """
    return structured_assistant.generate_response(
        mode,
        question,
        patient_data,
        conversation_history
    )


def get_available_modes() -> List[Tuple[str, str, str]]:
    """
    Get list of available modes with descriptions and icons
    
    Returns:
        List of (mode_id, description, icon)
    """
    modes = [
        (
            ClinicalAssistantMode.DIAGNOSIS_SUPPORT,
            structured_assistant.get_mode_description(ClinicalAssistantMode.DIAGNOSIS_SUPPORT),
            structured_assistant.get_mode_icon(ClinicalAssistantMode.DIAGNOSIS_SUPPORT)
        ),
        (
            ClinicalAssistantMode.TREATMENT_RECOMMENDATION,
            structured_assistant.get_mode_description(ClinicalAssistantMode.TREATMENT_RECOMMENDATION),
            structured_assistant.get_mode_icon(ClinicalAssistantMode.TREATMENT_RECOMMENDATION)
        ),
        (
            ClinicalAssistantMode.RISK_EXPLANATION,
            structured_assistant.get_mode_description(ClinicalAssistantMode.RISK_EXPLANATION),
            structured_assistant.get_mode_icon(ClinicalAssistantMode.RISK_EXPLANATION)
        ),
        (
            ClinicalAssistantMode.MONITORING_ANALYSIS,
            structured_assistant.get_mode_description(ClinicalAssistantMode.MONITORING_ANALYSIS),
            structured_assistant.get_mode_icon(ClinicalAssistantMode.MONITORING_ANALYSIS)
        ),
    ]
    return modes


if __name__ == "__main__":
    # Example usage
    sample_patient = {
        'Name': 'John Doe',
        'Age': 25,
        'Severity': 'Severe',
        'Mutation': 'Intron22',
        'Dose': 50,
        'Exposure': 20,
        'Family_History': 'Yes',
        'Previous_Inhibitor': 'No',
        'Risk': 0.65
    }
    
    question = "What does my high risk score mean and what should I do about it?"
    
    response, mode = get_clinical_response(
        ClinicalAssistantMode.RISK_EXPLANATION,
        question,
        sample_patient
    )
    
    print(f"Mode: {mode}")
    print(f"Response: {response}")
