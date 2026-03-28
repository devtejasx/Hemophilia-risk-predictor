import os
from openai import OpenAI
import json
from typing import Optional, List, Dict

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

SYSTEM_PROMPT = """You are Dr. Hemophilia, an expert AI medical assistant specializing in hemophilia and bleeding disorders. 
You provide comprehensive, evidence-based clinical information to healthcare professionals and patients.

Your expertise includes:
- Hemophilia A and B pathophysiology and genetics
- Inhibitor development and management
- Treatment modalities (prophylaxis, on-demand, novel therapies)
- Monitoring and risk stratification
- Clinical complications and their management
- Genetic counseling and inheritance patterns
- Quality of life considerations

IMPORTANT GUIDELINES:
1. Always provide evidence-based medical information
2. Include personalized recommendations based on patient data when available
3. Highlight important clinical decision points
4. Flag high-risk situations requiring immediate specialist attention
5. Suggest monitoring schedules and frequency appropriate to risk level
6. Explain complex concepts in understandable but medically accurate language
7. Always include disclaimers that this is supplementary to professional medical judgment
8. Be proactive in identifying concerning patterns or trends

When discussing specific patients, tailor recommendations to:
- Mutation type and associated risk
- Current severity classification
- Previous treatment response
- Comorbidities and immunological factors
- Social/lifestyle factors affecting adherence

CLINICAL DECISION SUPPORT:
- Risk stratification: Provide context for risk scores and what they mean clinically
- Treatment selection: Discuss options with pros/cons relative to patient profile
- Monitoring intensity: Recommend frequency based on risk and phase of disease
- Inhibitor management: Include latest guidelines and emerging therapies"""

def create_gpt_response(user_message: str, patient_context: Optional[Dict] = None, conversation_history: Optional[List] = None) -> str:
    """Generate response using GPT-4 with patient context"""
    
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    # Build patient context string
    patient_info = ""
    if patient_context:
        patient_info = f"""
PATIENT CONTEXT FOR THIS CONVERSATION:
- Name: {patient_context.get('Name', 'N/A')}
- Age: {patient_context.get('Age', 'N/A')} years
- Severity: {patient_context.get('Severity', 'N/A')}
- Mutation: {patient_context.get('Mutation', 'N/A')}
- Risk Score: {patient_context.get('Risk', 0):.1%}
- Treatment Dose: {patient_context.get('Dose', 'N/A')} units
- Exposure Days: {patient_context.get('Exposure', 'N/A')}
- Product Type: {patient_context.get('Product', 'N/A')}
- Treatment Adherence: {patient_context.get('Adherence', 'N/A')}%
- Family History: {patient_context.get('Family History', 'N/A')}
- Previous Inhibitor: {patient_context.get('Previous Inhibitor', 'N/A')}
- Joint Damage Score: {patient_context.get('Joint Damage', 'N/A')}
- Bleeding Episodes/Year: {patient_context.get('Bleeding Episodes', 'N/A')}
- Vaccination Status: {patient_context.get('Vaccination', 'N/A')}
- Comorbidities: {patient_context.get('Comorbidities', 'None')}
"""
    
    # Build messages for conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + patient_info}
    ]
    
    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            messages.append({"role": "user", "content": msg.get('user_message', '')})
            messages.append({"role": "assistant", "content": msg.get('gpt_response', '')})
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        # Fallback to gpt-3.5-turbo if gpt-4 not available
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                top_p=0.9
            )
            return response.choices[0].message.content
        except Exception as e2:
            return f"Error generating response: {str(e2)[:100]}. Please check your OpenAI API key and ensure you have sufficient credits."

def get_clinical_recommendations(patient_data: Dict) -> str:
    """Generate clinical recommendations for a specific patient"""
    
    prompt = f"""Based on this patient's profile, provide comprehensive clinical recommendations:
    
Patient: {patient_data.get('Name')}
Risk Score: {patient_data.get('Risk', 0):.1%}
Severity: {patient_data.get('Severity')}
Mutation: {patient_data.get('Mutation')}
Exposure Days: {patient_data.get('Exposure')}

Please include:
1. Risk stratification and interpretation
2. Recommended treatment approach
3. Monitoring schedule and intensity
4. Inhibitor prevention strategies
5. Lifestyle and adherence recommendations
6. Red flags requiring urgent attention
7. Follow-up and reassessment timeline"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=2500
        )
        return response.choices[0].message.content
    except:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=2500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Unable to generate recommendations: {str(e)[:100]}"

def analyze_monitoring_data(patient_data: Dict, monitoring_records: List[Dict]) -> str:
    """Analyze monitoring data and provide clinical insights"""
    
    monitoring_summary = json.dumps(monitoring_records, indent=2, default=str)
    
    prompt = f"""Analyze this patient's monitoring data and provide clinical insights:

Patient: {patient_data.get('Name')} - Risk Level: {patient_data.get('Risk', 0):.1%}

Monitoring Records:
{monitoring_summary}

Please provide:
1. Trends in the monitoring data
2. Any concerning findings
3. Recommendations for monitoring frequency adjustment
4. Need for specialist referral
5. Suggestions for treatment adjustment if indicated"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Unable to analyze monitoring data: {str(e)[:100]}"

def generate_inhibitor_risk_explanation(patient_data: Dict, risk_factors: Dict) -> str:
    """Generate detailed inhibitor risk explanation"""
    
    risk_factors_str = json.dumps(risk_factors, indent=2)
    
    prompt = f"""Provide a detailed clinical explanation of this patient's inhibitor risk:

Patient Profile:
- Name: {patient_data.get('Name')}
- Mutation: {patient_data.get('Mutation')}
- Severity: {patient_data.get('Severity')}
- Risk Score: {patient_data.get('Risk', 0):.1%}
- Exposure Days: {patient_data.get('Exposure')}
- Family History: {patient_data.get('Family History')}

Risk Factors Breakdown:
{risk_factors_str}

Please provide:
1. Explanation of each contributing risk factor
2. Relative importance of each factor
3. Strategies to mitigate each modifiable risk factor
4. Timeline for inhibitor development risk
5. Recommended preventive measures
6. Monitoring protocol specifics"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Unable to generate risk explanation: {str(e)[:100]}"
