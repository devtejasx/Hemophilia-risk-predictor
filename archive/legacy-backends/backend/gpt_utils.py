"""
GPT and clinical assistant utilities
Extracted from gpt_chatbot.py and clinical_assistant.py for FastAPI backend
"""

import os
from openai import OpenAI
from typing import Tuple, Dict, Optional, List, Any
from datetime import datetime


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


class ClinicalAssistantMode:
    """Clinical assistant modes"""
    DIAGNOSIS_SUPPORT = "diagnosis_support"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    RISK_EXPLANATION = "risk_explanation"
    MONITORING_ANALYSIS = "monitoring_analysis"


MEDICAL_DEFINITIONS = {
    "inhibitor": "Antibodies against factor VIII or IX that reduce treatment effectiveness",
    "intron 22 inversion": "Most common hemophilia A mutation, accounts for ~45% of severe cases",
    "factor VIII": "Clotting factor VIII, deficient in hemophilia A",
    "factor IX": "Clotting factor IX, deficient in hemophilia B",
    "dosing interval": "Time between treatment administrations",
    "half-life": "Time for factor level to decrease to 50% of initial level",
    "extended half-life": "Newer products with increased half-life (8-20 hours vs 8-12 hours traditional)",
    "prophylaxis": "Preventive treatment given regularly to prevent bleeding",
    "on-demand": "Treatment given when bleeding occurs",
    "immune tolerance induction": "Treatment to eliminate inhibitors",
    "emicizumab": "Bispecific monoclonal antibody for hemophilia A with inhibitors",
    "AT": "Activated thrombin time, used for factor monitoring",
    "PTT": "Partial thromboplastin time, coagulation test",
    "joint damage": "Arthropathy from repeated hemarthrosis",
    "hemarthrosis": "Bleeding into a joint",
    "recombinant": "Laboratory-produced clotting factors (not from donated blood)",
    "plasma-derived": "Clotting factors extracted from donated blood plasma",
}


class StructuredPromptTemplates:
    """Structured prompt templates for clinical modes"""
    
    @staticmethod
    def diagnosis_support(patient_context: Optional[Dict] = None, question: str = "") -> str:
        """Diagnosis support mode template"""
        context_str = _format_patient_context(patient_context)
        
        return f"""You are an AI clinical decision support assistant for hemophilia care specializing in diagnostic interpretation.

{context_str}

CLINICAL GUIDELINES FOR THIS MODE:
- Focus on interpreting clinical findings and symptoms
- Support differential diagnosis reasoning
- Consider hemophilia-specific presentations and complications
- Integrate patient history with current presentation
- Provide evidence-based diagnostic considerations
- Highlight red flags requiring immediate intervention

CLINICAL REASONING FRAMEWORK:
1. Symptom analysis: What is the primary clinical presentation?
2. Historical context: Prior episodes, current treatment status
3. Differential considerations: What else could cause these findings?
4. Test interpretation: What tests would help clarify diagnosis?
5. Risk stratification: What is the urgency level?

CRITICAL: AI-generated suggestions are for diagnosis support only and should NOT replace professional medical judgment. This is educational information to support clinical discussion, not medical advice. Always consult with qualified hemophilia specialists for final diagnostic and management decisions.

Patient Question: {question}

Provide diagnostic support response:"""
    
    @staticmethod
    def treatment_recommendation(patient_context: Optional[Dict] = None, question: str = "") -> str:
        """Treatment recommendation mode template"""
        context_str = _format_patient_context(patient_context)
        
        return f"""You are an AI clinical decision support assistant for hemophilia care specializing in treatment optimization.

{context_str}

CLINICAL GUIDELINES FOR THIS MODE:
- Evaluate treatment dosing and frequency appropriateness
- Consider treatment modality options (on-demand vs prophylaxis)
- Assess adherence barriers and support strategies
- Compare available product options (plasma-derived, recombinant, extended half-life)
- Address treatment-related complications
- Optimize quality of life while maintaining safety

TREATMENT OPTIMIZATION FRAMEWORK:
1. Current regimen assessment: Is the current approach meeting goals?
2. Efficacy evaluation: Are bleeding episodes controlled?
3. Safety review: Any adverse events or complications?
4. Adherence analysis: What barriers exist to compliance?
5. Alternative options: What other approaches should be considered?

CRITICAL: AI-generated suggestions are for treatment discussion only and should NOT replace professional medical judgment. All treatment recommendations require specialist consultation and informed patient consent. This is educational support, not medical advice.

Patient Question: {question}

Provide treatment recommendation response:"""
    
    @staticmethod
    def risk_explanation(patient_context: Optional[Dict] = None, question: str = "") -> str:
        """Risk explanation mode template"""
        context_str = _format_patient_context(patient_context)
        
        return f"""You are an AI clinical decision support assistant for hemophilia care specializing in risk assessment.

{context_str}

CLINICAL GUIDELINES FOR THIS MODE:
- Interpret inhibitor development risk factors
- Explain protective factors that may reduce risk
- Assess individual risk profiles in context
- Provide evidence for risk stratification
- Discuss risk modification strategies
- Communicate uncertainty appropriately

RISK ASSESSMENT FRAMEWORK:
1. Risk factor review: What factors increase/decrease risk?
2. Evidence synthesis: What does literature say about this profile?
3. Comparative analysis: How does this patient compare to others?
4. Modification potential: What interventions could reduce risk?
5. Monitoring implications: What surveillance is appropriate?

CRITICAL: AI-generated suggestions are for risk explanation education only and should NOT replace professional medical judgment. Individual risk predictions are probabilistic and require specialist interpretation. This is educational support, not medical risk assessment.

Patient Question: {question}

Provide risk explanation response:"""
    
    @staticmethod
    def monitoring_analysis(patient_context: Optional[Dict] = None, question: str = "") -> str:
        """Monitoring analysis mode template"""
        context_str = _format_patient_context(patient_context)
        
        return f"""You are an AI clinical decision support assistant for hemophilia care specializing in monitoring protocols.

{context_str}

CLINICAL GUIDELINES FOR THIS MODE:
- Recommend appropriate monitoring frequency based on risk
- Guide inhibitor screening protocols
- Establish surveillance milestones
- Define monitoring test selection
- Address special monitoring situations (acute events, treatment changes)
- Provide monitoring documentation guidance

MONITORING PROTOCOL FRAMEWORK:
1. Current status: What monitoring is currently being done?
2. Risk alignment: Does monitoring match risk level?
3. Frequency determination: How often should labs/tests occur?
4. Test selection: Which specific tests are needed?
5. Escalation triggers: What findings warrant intervention?

CRITICAL: AI-generated suggestions are for monitoring guidance only and should NOT replace professional medical judgment. Monitoring protocols require specialist oversight and adaptation to individual clinical changes. This is educational support, not definitive clinical guidance.

Patient Question: {question}

Provide monitoring analysis response:"""


def _format_patient_context(patient_data: Optional[Dict]) -> str:
    """Format patient data for prompt inclusion"""
    if not patient_data:
        return "No specific patient data loaded. Respond in general education mode."
    
    # Extract key fields
    lines = ["PATIENT CONTEXT:"]
    
    if patient_data.get('Name'):
        lines.append(f"- Patient: {patient_data['Name']}, {patient_data.get('Age', '?')} years old")
    
    if patient_data.get('Severity'):
        lines.append(f"- Severity: {patient_data['Severity']}")
    
    if patient_data.get('Mutation'):
        lines.append(f"- Mutation: {patient_data['Mutation']}")
    
    if patient_data.get('Dose'):
        lines.append(f"- Treatment: {patient_data['Dose']} units, {patient_data.get('Product', 'unspecified')}, {patient_data.get('Exposure', '?')} days exposure")
    
    if patient_data.get('Risk') is not None:
        lines.append(f"- Inhibitor Risk: {patient_data['Risk']:.1%}")
    
    if patient_data.get('Bleeding Episodes'):
        lines.append(f"- Bleeding Episodes/Month: {patient_data['Bleeding Episodes']}")
    
    if patient_data.get('Joint Damage'):
        lines.append(f"- Joint Damage: {patient_data['Joint Damage']} affected joints")
    
    if patient_data.get('Previous Inhibitor'):
        lines.append(f"- Previous Inhibitor: Yes - INCREASED RISK")
    
    if patient_data.get('Adherence'):
        lines.append(f"- Adherence: {patient_data['Adherence']:.0%}")
    
    return "\n".join(lines)


def get_clinical_response(
    mode: str,
    question: str,
    patient_data: Optional[Dict] = None,
    conversation_history: Optional[List[Dict]] = None
) -> Tuple[str, str]:
    """
    Generate clinical AI response using structured modes
    
    Returns: (response_text, mode_used)
    """
    if conversation_history is None:
        conversation_history = []
    
    # Build system prompt based on mode
    if mode == ClinicalAssistantMode.DIAGNOSIS_SUPPORT:
        system_prompt = StructuredPromptTemplates.diagnosis_support(patient_data, question)
    elif mode == ClinicalAssistantMode.TREATMENT_RECOMMENDATION:
        system_prompt = StructuredPromptTemplates.treatment_recommendation(patient_data, question)
    elif mode == ClinicalAssistantMode.RISK_EXPLANATION:
        system_prompt = StructuredPromptTemplates.risk_explanation(patient_data, question)
    elif mode == ClinicalAssistantMode.MONITORING_ANALYSIS:
        system_prompt = StructuredPromptTemplates.monitoring_analysis(patient_data, question)
    else:
        system_prompt = StructuredPromptTemplates.diagnosis_support(patient_data, question)
    
    # Build messages for API call
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add conversation history (last 5 messages for context)
    for msg in conversation_history[-5:]:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })
    
    # Add current question
    messages.append({"role": "user", "content": question})
    
    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9
        )
        
        return response.choices[0].message.content, mode
        
    except Exception as e:
        error_msg = str(e)
        
        if "401" in error_msg or "invalid_request_error" in error_msg.lower():
            return "Error: Invalid or missing OpenAI API key. Please set OPENAI_API_KEY environment variable.", mode
        elif "429" in error_msg:
            return "Error: API rate limit exceeded. Please try again in a moment.", mode
        else:
            return f"Error generating response: {error_msg[:100]}", mode


def get_available_modes() -> List[Tuple[str, str, str]]:
    """
    Get list of available modes with descriptions and icons
    
    Returns: list of (mode_id, description, icon)
    """
    return [
        (ClinicalAssistantMode.DIAGNOSIS_SUPPORT, 
         "Interpret symptoms and findings", 
         "🔍"),
        (ClinicalAssistantMode.TREATMENT_RECOMMENDATION, 
         "Optimize treatment strategies", 
         "💊"),
        (ClinicalAssistantMode.RISK_EXPLANATION, 
         "Explain risk factors and scores", 
         "⚠️"),
        (ClinicalAssistantMode.MONITORING_ANALYSIS, 
         "Guide monitoring protocols", 
         "📊"),
    ]


def get_medical_definitions(terms: Optional[List[str]] = None) -> Dict[str, str]:
    """Get medical terminology definitions"""
    if terms:
        return {
            term.lower(): definition 
            for term, definition in MEDICAL_DEFINITIONS.items()
            if term.lower() in [t.lower() for t in terms]
        }
    return MEDICAL_DEFINITIONS


def create_gpt_response(
    question: str,
    patient_context: Optional[Dict] = None,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Legacy GPT response function (for backward compatibility)
    Uses default diagnosis support mode
    """
    response, _ = get_clinical_response(
        ClinicalAssistantMode.DIAGNOSIS_SUPPORT,
        question,
        patient_context,
        conversation_history
    )
    return response
