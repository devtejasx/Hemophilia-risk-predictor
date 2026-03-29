import os
from openai import OpenAI
import json
from typing import Optional, List, Dict
import pandas as pd
import sqlite3
from local_model import get_local_response, LocalChatbot

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Initialize local chatbot (pre-trained model)
local_chatbot = None
try:
    local_chatbot = LocalChatbot()
    print("✅ Local chatbot initialized successfully")
except Exception as e:
    print(f"⚠️ Local chatbot initialization: {e}")

# Database connectivity for learning from patients
def get_similar_patients(patient_data: Dict, limit: int = 5) -> List[Dict]:
    """Find similar patients from database to learn from their experiences"""
    try:
        conn = sqlite3.connect("hemophilia_clinic.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Find patients with similar characteristics
        mutation = patient_data.get('Mutation', '')
        severity = patient_data.get('Severity', '')
        risk_score = float(patient_data.get('Risk', 0)) if isinstance(patient_data.get('Risk'), (int, float)) else 0
        
        query = """
            SELECT id, name, age, severity, mutation, dose, exposure, 
                   risk_score, treatment_adherence, previous_inhibitor
            FROM patients 
            WHERE severity = ? AND mutation LIKE ?
            ORDER BY ABS(risk_score - ?) ASC
            LIMIT ?
        """
        
        cursor.execute(query, (severity, f"%{mutation}%", risk_score, limit))
        similar = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return similar
    except Exception as e:
        return []

def get_patient_statistics(mutation: str = None, severity: str = None) -> Dict:
    """Get aggregated statistics about patients for insights"""
    try:
        conn = sqlite3.connect("hemophilia_clinic.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        stats = {}
        
        # Overall stats
        cursor.execute("SELECT COUNT(*) as total, AVG(risk_score) as avg_risk, AVG(treatment_adherence) as avg_adherence FROM patients")
        overall = dict(cursor.fetchone())
        stats['overall'] = overall
        
        # By mutation
        if mutation:
            cursor.execute(
                "SELECT COUNT(*) as count, AVG(risk_score) as avg_risk FROM patients WHERE mutation LIKE ?",
                (f"%{mutation}%",)
            )
            stats['by_mutation'] = dict(cursor.fetchone())
        
        # By severity
        if severity:
            cursor.execute(
                "SELECT COUNT(*) as count, AVG(risk_score) as avg_risk FROM patients WHERE severity = ?",
                (severity,)
            )
            stats['by_severity'] = dict(cursor.fetchone())
        
        # Inhibitor statistics
        cursor.execute("SELECT previous_inhibitor, COUNT(*) as count FROM patients GROUP BY previous_inhibitor")
        inhibitor_stats = {row['previous_inhibitor']: row['count'] for row in cursor.fetchall()}
        stats['inhibitor_outcomes'] = inhibitor_stats
        
        conn.close()
        return stats
    except Exception as e:
        return {}

def get_treatment_patterns(mutation: str = None, severity: str = None) -> str:
    """Learn treatment patterns from similar patients"""
    try:
        conn = sqlite3.connect("hemophilia_clinic.db")
        cursor = conn.cursor()
        
        if mutation and severity:
            query = """
                SELECT dose, exposure, treatment_adherence, risk_score, previous_inhibitor
                FROM patients
                WHERE mutation LIKE ? AND severity = ?
                LIMIT 10
            """
            cursor.execute(query, (f"%{mutation}%", severity))
        elif mutation:
            query = """
                SELECT dose, exposure, treatment_adherence, risk_score, previous_inhibitor
                FROM patients
                WHERE mutation LIKE ?
                LIMIT 10
            """
            cursor.execute(query, (f"%{mutation}%",))
        else:
            query = "SELECT dose, exposure, treatment_adherence, risk_score, previous_inhibitor FROM patients LIMIT 10"
            cursor.execute(query)
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            doses = [r[0] for r in results]
            adherences = [r[2] for r in results]
            
            avg_dose = sum(doses) / len(doses) if doses else 0
            avg_adherence = sum(adherences) / len(adherences) if adherences else 0
            
            return f"Similar patients: avg dose {avg_dose:.0f} units, avg adherence {avg_adherence:.0f}%"
        
        return ""
    except Exception as e:
        return ""

def record_patient_feedback(patient_name: str, interaction_quality: str, notes: str = "") -> bool:
    """Record feedback from chatbot interactions to continuously improve"""
    try:
        conn = sqlite3.connect("hemophilia_clinic.db")
        cursor = conn.cursor()
        
        # Create feedback table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chatbot_feedback (
                id INTEGER PRIMARY KEY,
                patient_name TEXT,
                interaction_quality TEXT,
                notes TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute(
            "INSERT INTO chatbot_feedback (patient_name, interaction_quality, notes) VALUES (?, ?, ?)",
            (patient_name, interaction_quality, notes)
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

SYSTEM_PROMPT = """You are a helpful AI medical assistant with expertise in hemophilia and general medical knowledge.

YOUR PRIMARY ROLE:
1. Help patients understand their hemophilia condition and risk factors
2. Answer ANY question the user asks - whether medical or general knowledge
3. Provide personalized recommendations when patient data is available
4. Explain complex concepts in simple, easy-to-understand language

WHEN ANSWERING QUESTIONS:
- If it's about the patient's specific condition, tailor your response to THEIR profile
- If it's a general question (hemophilia, health, lifestyle, etc.), provide helpful accurate information
- If it's completely unrelated to medical topics, politely answer but try to connect back to their health if relevant
- Always be friendly, conversational, and use simple English
- Break complex info into bullet points when helpful

IMPORTANT:
- Never refuse to answer reasonable questions
- Always provide accurate, helpful information
- If unsure, say so and suggest consulting their doctor
- Use their patient data to personalize medical advice
- Be encouraging and supportive

You are an intelligent assistant who can discuss:
✅ Hemophilia and bleeding disorders
✅ Treatment options and lifestyle
✅ Health and wellness topics
✅ General knowledge questions
✅ Personal health concerns
✅ ANY question the user asks in simple English"""

def create_gpt_response(user_message: str, patient_context: Optional[Dict] = None, conversation_history: Optional[List] = None) -> str:
    """Generate response using GPT-4 with patient context"""
    
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    # Build patient context string
    patient_info = ""
    database_insights = ""
    
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
        
        # Fetch similar patients and patterns from database for additional context
        similar_patients = get_similar_patients(patient_context, limit=3)
        treatment_patterns = get_treatment_patterns(
            patient_context.get('Mutation', ''),
            patient_context.get('Severity', '')
        )
        stats = get_patient_statistics(
            patient_context.get('Mutation', ''),
            patient_context.get('Severity', '')
        )
        
        database_insights = "\n\nLEARNING FROM SYSTEM DATA:"
        
        if similar_patients:
            database_insights += f"\nSimilar patients in system: {len(similar_patients)} found"
            for p in similar_patients:
                risk_score = p.get('risk_score')
                dose = p.get('dose', 'N/A')
                name = p.get('name', 'Unknown')
                # Safe formatting - handle None values
                risk_str = f"{float(risk_score):.1%}" if risk_score is not None else "N/A"
                database_insights += f"\n  - {name}: Risk {risk_str}, Dose {dose} units"
        
        if treatment_patterns:
            database_insights += f"\n\nTreatment patterns from similar patients: {treatment_patterns}"
        
        if stats.get('by_mutation') and stats['by_mutation']:
            count = stats['by_mutation'].get('count', 0)
            avg_risk = stats['by_mutation'].get('avg_risk', 0)
            # Safe formatting for avg_risk
            avg_risk_str = f"{float(avg_risk):.1%}" if avg_risk is not None else "N/A"
            database_insights += f"\nMutation statistics: {count} patients, avg risk {avg_risk_str}"
        
        if stats.get('overall') and stats['overall']:
            total = stats['overall'].get('total', 0)
            database_insights += f"\nSystem-wide: {total} patients tracked"
    
    # Build messages for conversation
    system_prompt_with_context = SYSTEM_PROMPT + patient_info + database_insights
    
    messages = [
        {"role": "system", "content": system_prompt_with_context}
    ]
    
    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            if msg.get('role') and msg.get('content'):
                messages.append({"role": msg.get('role'), "content": msg.get('content')})
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    # Try local model first - it works offline and is fast!
    try:
        if local_chatbot:
            local_response = local_chatbot.generate_response(
                user_message,
                context=patient_context,
                max_length=300
            )
            if local_response:
                return local_response
    except Exception as e:
        print(f"Local model failed, trying API: {e}")
    
    # Fallback to GPT API if local model fails or doesn't respond
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
            # If all APIs fail, use smart fallback response
            return generate_fallback_response(user_message, patient_context)

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

def generate_fallback_response(user_message: str, patient_data: Optional[Dict] = None) -> str:
    """Generate response using clinical knowledge base + database learning when API is unavailable"""
    
    user_msg_lower = user_message.lower()
    
    # Build patient context
    patient_severity = patient_data.get('Severity', 'N/A') if patient_data else 'N/A'
    patient_mutation = patient_data.get('Mutation', 'N/A') if patient_data else 'N/A'
    patient_risk = patient_data.get('Risk', 0) if patient_data else 0
    patient_name = patient_data.get('Name', 'Patient') if patient_data else 'Patient'
    patient_dose = patient_data.get('Dose', 'N/A') if patient_data else 'N/A'
    
    # Get database insights
    similar_patients = get_similar_patients(patient_data or {}, limit=5) if patient_data else []
    treatment_patterns = get_treatment_patterns(patient_mutation, patient_severity) if patient_data else ""
    system_stats = get_patient_statistics(patient_mutation, patient_severity) if patient_data else {}
    
    # Build database insights section
    db_insights = ""
    if similar_patients:
        db_insights += f"\n**📊 System Data:** Found {len(similar_patients)} similar patients in our database being tracked"
    
    if system_stats.get('by_mutation') and system_stats['by_mutation'].get('count', 0) > 0:
        count = system_stats['by_mutation']['count']
        avg_risk = system_stats['by_mutation'].get('avg_risk')
        # Safe formatting
        avg_risk_str = f"{float(avg_risk):.1%}" if avg_risk is not None else "N/A"
        db_insights += f"\n**Patients with {patient_mutation}:** {count} cases tracked, avg risk {avg_risk_str}"
    
    # Risk level emoji
    if patient_risk > 0.8:
        risk_level = "🔴 CRITICAL"
    elif patient_risk > 0.6:
        risk_level = "🟠 HIGH"
    elif patient_risk > 0.4:
        risk_level = "🟡 MODERATE"
    else:
        risk_level = "🟢 LOW"
    
    # Knowledge base responses
    responses = {
        "mutation": f"""
**About Your Mutation: {patient_mutation}**

Hemophilia mutations vary widely. Your mutation type ({patient_mutation}) affects:
- The severity of your bleeding disorder
- How often you need treatment
- Your inhibitor development risk
- Your family planning considerations

**Your Current Status:**
- Severity: {patient_severity}
- Current Risk Level: {risk_level} ({patient_risk:.1%})

**Next Steps:** 
Discuss your specific mutation details with your hemophilia treatment team for personalized guidance based on the latest research.
""",
        
        "risk": f"""
**Your Inhibitor Risk Assessment**

Your current risk score: {risk_level} ({patient_risk:.1%})

**What This Means:**
- Below 40%: Low risk - continue current monitoring protocol
- 40-60%: Moderate risk - monthly inhibitor screening recommended
- 60-80%: High risk - frequent monitoring and aggressive inhibitor prevention
- Above 80%: Critical risk - intensive monitoring, consider treatment changes

**Your Risk Level:** {risk_level}

**Recommendations Based on Your Risk:**
1. Regular inhibitor screening as per your team's protocol
2. Maintain treatment adherence for best protection
3. Discuss inhibitor prevention strategies with your team
4. Report any unusual bleeding patterns immediately

{db_insights}

*This assessment is based on clinical factors in your profile. Your treatment team has the full clinical context.*
""",
        
        "treatment": f"""
**Your Treatment Plan**

**Current Configuration:**
- Severity: {patient_severity}
- Dose: {patient_dose} units
- Product: {patient_data.get('Product', 'N/A') if patient_data else 'N/A'}
- Adherence: {patient_data.get('Treatment_Adherence', 'N/A') if patient_data else 'N/A'}%

{treatment_patterns}

**General Treatment Options for {patient_severity} Hemophilia:**
1. **Prophylaxis** - Regular preventive dosing to maintain adequate factor levels
2. **On-Demand** - Treatment only when bleeding occurs
3. **Extended Half-Life (EHL)** - Longer-lasting factor products
4. **Novel Therapies** - Bispecific antibodies and other emerging treatments

**Discussion Points with Your Team:**
- Is current dosing adequate for your lifestyle?
- Have you considered alternative factor products?
- Are there new therapies appropriate for you?

*Your treatment should be individualized by your hemophilia team.*
""",
        
        "monitoring": f"""
**Your Monitoring Schedule**

Based on your {patient_severity} severity and {risk_level} risk level:

**Recommended Monitoring:**
- Clinical visits: {3 if patient_risk > 0.6 else 2 if patient_risk > 0.4 else 1} per year minimum
- Inhibitor screening: {'Monthly' if patient_risk > 0.6 else 'Quarterly' if patient_risk > 0.4 else 'As clinically indicated'}
- Joint assessments: Annually
- Viral serology: As per latest guidelines

**Important Tests:**
- Factor level measurement
- Inhibitor titers (if previously positive)
- Liver function tests
- Joint ultrasound/imaging (as indicated)

**When to Contact Your Team Urgently:**
- Unusual or prolonged bleeding
- Swelling or pain in joints
- Fever or signs of infection
- Questions about changes in your condition

*Your specific schedule should be determined by your hemophilia treatment center.*
""",
        
        "activity": f"""
**Physical Activity and Exercise**

For {patient_severity} Hemophilia with {risk_level} risk:

**Generally Safe Activities:**
- Walking and controlled aerobic exercise
- Swimming (high factor coverage)
- Cycling with protective gear
- Strength training with proper form

**Activities to Discuss with Your Team First:**
- Contact sports (basketball, soccer, football)
- High-impact activities (skiing, skateboarding)
- Intense weight lifting
- Martial arts

**General Safety Tips:**
1. Ensure adequate factor coverage before activity
2. Use appropriate protective equipment
3. Start gradually with new activities
4. Keep your team informed of regular exercise plans
5. Have an emergency plan in place

**Benefits of Activity:**
- Improved joint health
- Better physical fitness
- Reduced anxiety
- Enhanced quality of life

*Always discuss your exercise plan with your hemophilia team.*
""",
        
        "inhibitor": f"""
**Inhibitor Development and Prevention**

Your current inhibitor risk: {risk_level}

**What is an Inhibitor?**
An inhibitor is an antibody that neutralizes clotting factor, making it less effective. This is a serious complication in hemophilia.

**Risk Factors for Inhibitor Development:**
- Early treatment exposure
- Severe hemophilia
- Certain mutations
- Non-adherence to treatment
- Immune challenges

**Prevention Strategies:**
1. Maintain consistent factor product use
2. Follow your treatment protocol precisely
3. Regular inhibitor screening
4. Report any changes in treatment response
5. Consider inhibitor-sparing strategies if appropriate

**If You Develop an Inhibitor:**
- Increased monitoring is needed
- Alternative treatment strategies exist
- Immune tolerance induction (ITI) may be considered
- Specialized team consultation required

**Your Status:** Regular monitoring recommended given your {risk_level} risk level.

*Discuss inhibitor prevention strategies specific to your situation with your team.*
""",
        
        "warning": f"""
**Warning Signs - When to Seek Help**

**Seek Immediate Medical Attention For:**
🚨 Severe headaches or head injuries
🚨 Difficulty swallowing or breathing
🚨 Severe abdominal pain
🚨 Uncontrollable bleeding
🚨 Severe chest pain
🚨 Signs of stroke or neurological changes

**Contact Your Team Urgently For:**
⚠️ Unusual or excessive bruising
⚠️ Joint pain or swelling
⚠️ Muscle pain or swelling
⚠️ Blood in urine or stool
⚠️ Nosebleeds lasting >30 minutes
⚠️ Mouth bleeding
⚠️ Fever or signs of infection

**Routine Follow-up For:**
📋 Questions about dosing
📋 Medication side effects
📋 Adherence difficulties
📋 Lifestyle adjustments

**Emergency Contact:**
Keep your treatment center's emergency number readily available.

*Given your {risk_level} risk level, be especially vigilant with monitoring.*
""",
        
        "pregnancy": f"""
**Family Planning and Pregnancy**

**For Females with Hemophilia:**
- Genetic counseling is important before planning pregnancy
- Factor levels change during pregnancy
- Delivery planning needs specialized hematology/OB coordination
- Carrier daughters may be mildly affected
- Genetic testing of partner may be recommended

**For Males and Carriers:**
- Genetic counseling recommended
- All sons of affected males will be carriers
- Daughters may be affected depending on partner's genetics
- Preconception planning valuable

**Considerations:**
- Discuss with your hemophilia team and OB/GYN
- Consider genetic counseling
- Treatment will likely be needed during pregnancy/delivery
- Specialized centers can support optimized care

**Next Steps:**
1. Meet with hemophilia treatment team
2. Consider genetic counseling referral
3. Coordinate with OB/GYN
4. Discuss all medication safety concerns

*Pregnancy can be safe with proper planning and specialist coordination.*
""",
    }
    
    # Match user question to response
    default_response = f"""
**Clinical Assistant Response**

I'm functioning in offline mode due to API limitations. While I can provide general hemophilia information, **I recommend discussing this with your hemophilia treatment team for personalized medical advice.**

**About {patient_name}'s Profile:**
- Severity: {patient_severity}
- Current Risk Level: {risk_level} ({patient_risk:.1%})
- Mutation: {patient_mutation}

**Your Question:** "{user_message}"

**General Information:**
Hemophilia management is highly individualized. The best guidance comes from your treatment team who knows your complete medical history, current status, and latest recommendations.

**How to Get Answers:**
1. Contact your hemophilia treatment center
2. Speak with your hematologist
3. Reach out to patient organizations (NHF, WFH)
4. Consult nursing coordinators

**I can help with:**
✅ General hemophilia education
✅ Explaining your risk profile
✅ Suggesting discussion points for your care team
✅ Providing monitoring guidelines

*Would you like information about any of these topics: mutation, risk, treatment, monitoring, activity, inhibitors, warning signs, pregnancy planning?*
"""
    
    # Check keywords in user message for hemophilia-specific questions
    for keyword, response in responses.items():
        if keyword in user_msg_lower:
            return response
    
    # For any other general questions, provide a helpful response
    return generate_general_answer(user_message, patient_data, patient_severity, patient_risk)

def generate_general_answer(question: str, patient_data: Optional[Dict], severity: str, risk: float) -> str:
    """Generate answer for any general question the user asks"""
    
    question_lower = question.lower()
    
    # Simple Q&A system for common general questions
    general_responses = {
        "hello": "👋 Hi there! I'm your AI health assistant. I'm here to help with questions about hemophilia, health, or anything else you'd like to know. What can I help you with?",
        
        "how are you": "I'm functioning well, thank you for asking! 😊 More importantly, how are YOU doing? Is there anything health-related I can help you with?",
        
        "what is hemophilia": f"""
**What is Hemophilia?**

Hemophilia is a bleeding disorder where your blood doesn't clot properly. There are two main types:
- **Hemophilia A**: Missing clotting factor VIII
- **Hemophilia B**: Missing clotting factor IX

**Key Facts:**
- It's genetic and usually inherited
- Can range from mild to severe
- Managed with factor replacement therapy
- With proper treatment, people can live normal lives
- {f"Your current status: {severity} severity, {risk:.1%} inhibitor risk" if patient_data else ""}
""",
        
        "what should i eat": f"""
**Nutrition Tips for You**

**Generally Good Foods:**
- Lean proteins (chicken, fish, turkey)
- Fruits and vegetables (rich in vitamins)
- Whole grains and fiber
- Dairy products (calcium for bones)
- Iron-rich foods (spinach, beans, red meat)

**Important Considerations:**
- Avoid excessive alcohol (increases bleeding risk)
- Stay hydrated
- Maintain healthy weight
- Avoid fatty foods in excess

{f"**Your Profile:** With {severity} hemophilia and {risk:.1%} risk level, proper nutrition supports treatment effectiveness." if patient_data else ""}

**Tip:** Consult a nutritionist for personalized dietary advice.
""",
        
        "exercise": f"""
**Exercise & Physical Activity**

**Safe Activities:**
✅ Walking and jogging
✅ Swimming (excellent option)
✅ Cycling
✅ Low-impact aerobics
✅ Yoga and stretching
✅ Bowling

**Avoid or Discuss First:**
⚠️ Contact sports (basketball, soccer, football)
⚠️ Heavy weightlifting
⚠️ High-impact activities (jumping, skateboarding)
⚠️ Martial arts

**General Tips:**
- Warm up before exercising
- Stay well-hydrated
- Wear protective gear
- Have factor on hand
- Start gradually with new activities

{f"**Your Status:** With {severity} hemophilia, stick to lower-impact activities unless cleared by your doctor." if patient_data else ""}
""",
        
        "stress": """
**Managing Stress**

**Helpful Techniques:**
- Deep breathing exercises
- Meditation or mindfulness
- Regular exercise
- Hobbies and recreational activities
- Talking with friends/family
- Professional counseling if needed

**Why It Matters:**
Stress can affect your adherence and overall health. Managing it is important for your well-being.
""",
        
        "sleep": """
**Sleep & Rest**

**Tips for Better Sleep:**
- Keep consistent sleep schedule
- Avoid screens 1 hour before bed
- Keep bedroom cool and dark
- Avoid caffeine late in day
- Exercise regularly (but not before bed)
- Relax before sleep

**Target:** 7-9 hours per night for adults
""",
        
        "travel": f"""
**Traveling with Hemophilia**

**Important Steps:**
✅ Carry factor treatment with you
✅ Keep medical letter from your doctor
✅ Pack in carry-on luggage
✅ Research hospitals at destination
✅ Store properly (temperature controlled)
✅ Know emergency numbers

**Planning:**
- Notify your doctor before traveling
- Get comprehensive travel insurance
- Keep medications in original packaging
- Travel with someone who knows your condition

{f"**Your Consideration:** With {risk:.1%} risk level, ensure access to emergency care." if patient_data else ""}
""",
    }
    
    # Check for keyword matches
    for keyword, response in general_responses.items():
        if keyword in question_lower:
            return response
    
    # Default helpful response for any other question
    return f"""
**I'm Here to Help!** 👋

You asked: "{question}"

I'm an AI health assistant that can help with many topics. While I aim to be helpful, I work best with questions about:
- 🩸 Hemophilia management and treatment
- 💊 Health and wellness topics
- 🏃 Lifestyle and activities
- 📊 Your personal health data
- ❓ General knowledge questions

**For Your Question:**
If this is a medical question specific to YOUR condition, I'd recommend discussing it with your hemophilia treatment team or doctor for personalized advice.

**Try asking me about:**
- Your mutation or risk level
- Treatment options for people like you
- Safe activities and exercises
- What to eat or how to manage stress
- Warning signs to watch for
- Monitoring schedules

What would you like to know more about? 😊
"""

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


# ============ ADVANCED CHATGPT-LEVEL SMART ASSISTANCE ============

def analyze_case_complexity(patient_data: Dict, clinical_history: List[Dict]) -> str:
    """Advanced case complexity analysis with clinical reasoning"""
    
    history_str = json.dumps(clinical_history, indent=2, default=str) if clinical_history else "No previous history"
    
    prompt = f"""Perform an advanced clinical reasoning analysis for this complex case:

Current Patient Profile:
- Name: {patient_data.get('Name')}
- Age: {patient_data.get('Age')}
- Mutation: {patient_data.get('Mutation')}
- Severity: {patient_data.get('Severity')}
- Risk Score: {patient_data.get('Risk', 0):.1%}
- Current Adherence: {patient_data.get('Adherence', 'N/A')}%
- Joint Damage: {patient_data.get('Joint Damage', 'N/A')}

Clinical History:
{history_str}

Provide a comprehensive analysis including:
1. Case complexity assessment (simple, moderate, complex, highly complex)
2. Key clinical decision points
3. Differential considerations
4. Potential complications and how to prevent them
5. Evidence-based treatment optimization recommendations
6. Patient-specific risk mitigation strategies
7. When to escalate to specialist care
8. Long-term prognostic assessment"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.8,
            max_tokens=3000
        )
        return response.choices[0].message.content
    except:
        return "Advanced case analysis unavailable. Please check API configuration."

def generate_treatment_plan(patient_data: Dict) -> str:
    """Generate personalized comprehensive treatment plan"""
    
    prompt = f"""Create a detailed, personalized treatment plan for this patient:

Patient: {patient_data.get('Name')} (Age: {patient_data.get('Age')})
Mutation: {patient_data.get('Mutation')}
Severity: {patient_data.get('Severity')}
Risk Score: {patient_data.get('Risk', 0):.1%}
Adherence: {patient_data.get('Adherence', 'N/A')}%
Family History: {patient_data.get('Family History')}

Create a comprehensive plan covering:
1. Immediate treatment recommendations (0-1 month)
2. Short-term goals (1-3 months)
3. Long-term management strategy (3-12 months)
4. Prophylaxis vs on-demand considerations
5. Product selection rationale
6. Dosing optimization strategy
7. Monitoring schedule with specific frequency
8. Adherence enhancement strategies
9. Patient education priorities
10. Family and social support considerations
11. Complications prevention plan
12. When to reassess and adjust plan"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=3500
        )
        return response.choices[0].message.content
    except:
        return "Treatment plan generation unavailable. Please check API configuration."

def compare_treatment_options(patient_data: Dict, treatment_options: List[str]) -> str:
    """Advanced comparison of treatment options for patient"""
    
    options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(treatment_options)])
    
    prompt = f"""Provide a sophisticated clinical comparison of treatment options:

Patient Profile:
- Mutation: {patient_data.get('Mutation')}
- Severity: {patient_data.get('Severity')}
- Exposure Days: {patient_data.get('Exposure')}
- Risk Score: {patient_data.get('Risk', 0):.1%}
- Age: {patient_data.get('Age')}
- Adherence: {patient_data.get('Adherence', 'N/A')}%

Treatment Options:
{options_str}

Compare these options including:
1. Efficacy and effectiveness for this patient
2. Safety profile and contraindications
3. Feasibility in patient's context
4. Cost-effectiveness
5. Impact on quality of life
6. Long-term outcomes and sustainability
7. Monitoring requirements
8. Adverse event risks
9. Patient preference considerations
10. Recommendation with clinical rationale"""
    
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
        return "Treatment comparison analysis unavailable."

def explain_test_results(test_name: str, result_value: float, reference_range: str, patient_data: Dict) -> str:
    """Explain clinical significance of test results in context"""
    
    prompt = f"""Provide expert clinical interpretation of this lab test:

Patient: {patient_data.get('Name')} (Mutation: {patient_data.get('Mutation')}, Severity: {patient_data.get('Severity')})
Test: {test_name}
Result: {result_value}
Reference Range: {reference_range}

Explain:
1. Whether the result is normal or abnormal
2. Clinical significance for this patient
3. Relationship to hemophilia management
4. Implications for treatment decisions
5. Need for follow-up testing
6. Actionable next steps
7. When this warrants specialist referral"""
    
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
        return "Test result interpretation unavailable."

def identify_clinical_alerts(patient_data: Dict, recent_data: Dict) -> List[Dict]:
    """Identify clinically significant alerts and safety concerns"""
    
    prompt = f"""Identify all clinically significant alerts for this patient:

Patient: {patient_data.get('Name')}
Risk Score: {patient_data.get('Risk', 0):.1%}
Mutation: {patient_data.get('Mutation')}
Severity: {patient_data.get('Severity')}

Recent Data:
{json.dumps(recent_data, indent=2, default=str)}

Identify and categorize alerts as:
CRITICAL (immediate action required)
HIGH (requires attention within 24-48 hours)
MODERATE (requires attention within 1 week)
LOW (informational)

For each alert, provide:
- Alert name
- Severity level
- Clinical reason
- Recommended action
- Timeline for action"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.8,
            max_tokens=2000
        )
        
        # Parse response to extract alerts
        return {
            "alerts": response.choices[0].message.content,
            "generated_at": datetime.now().isoformat() if 'datetime' in dir() else None
        }
    except:
        return {"alerts": "Alert system unavailable.", "generated_at": None}

def provide_patient_education(topic: str, patient_context: Dict) -> str:
    """Provide patient-friendly education on health topics"""
    
    prompt = f"""Create patient-friendly educational material about: {topic}
    
Patient Context:
- Age: {patient_context.get('Age')}
- Readability Level: Adult (adjust complexity as needed)
- Condition: Hemophilia {patient_context.get('Severity')}

Create clear, empowering education that:
1. Explains the topic in patient-friendly language
2. Addresses common misconceptions
3. Provides practical tips and strategies
4. Includes safety considerations
5. Gives specific examples
6. Encourages questions and engagement
7. Suggests when to contact healthcare provider"""
    
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
        return "Patient education material unavailable."

def multi_turn_consultation(messages: List[Dict]) -> str:
    """Advanced multi-turn consultation with contextual memory"""
    
    # Build comprehensive message history
    system_msg = {"role": "system", "content": SYSTEM_PROMPT + "\n\nYou are now in an advanced multi-turn consultation mode. Maintain context across messages and provide increasingly sophisticated analysis as you understand the case better."}
    
    all_messages = [system_msg] + messages
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=all_messages,
            temperature=0.75,
            max_tokens=2500
        )
        return response.choices[0].message.content
    except:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=all_messages,
                temperature=0.75,
                max_tokens=2500
            )
            return response.choices[0].message.content
        except:
            return "Multi-turn consultation unavailable."

def generate_progress_summary(patient_data: Dict, trend_data: List[Dict]) -> str:
    """Generate comprehensive progress summary and clinical outcomes"""
    
    trends_str = json.dumps(trend_data, indent=2, default=str) if trend_data else "No trend data available"
    
    prompt = f"""Generate a comprehensive progress summary and outcomes assessment:

Patient: {patient_data.get('Name')}
Initial Risk Score: {patient_data.get('Risk', 0):.1%}
Current Risk Score: {patient_data.get('Current Risk', patient_data.get('Risk', 0)):.1%}

Trend Data:
{trends_str}

Provide:
1. Overall progress assessment
2. Clinical milestones achieved
3. Adherence trends and impact
4. Complication or safety events
5. Treatment optimization successes
6. Remaining challenges
7. Adjusted care plan based on progress
8. Projected outcomes at 6 and 12 months
9. Recommendations for continuation or modification of approach"""
    
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
        return "Progress summary generation unavailable."

# Import datetime for timestamps
from datetime import datetime
