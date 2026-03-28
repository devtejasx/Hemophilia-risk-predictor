"""
Example usage of Hemophilia AI Platform modules
Demonstrates database and GPT chatbot integration
"""

import os

# Try to load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

from database import (
    init_database, add_patient, get_patient, get_all_patients,
    add_conversation, get_conversation_history, add_doctor_note,
    get_doctor_notes, get_dashboard_stats, search_patients
)
from gpt_chatbot import (
    create_gpt_response, get_clinical_recommendations,
    analyze_monitoring_data, generate_inhibitor_risk_explanation
)

def example_1_initialize_database():
    """Initialize database on first run"""
    print("=" * 50)
    print("Example 1: Initialize Database")
    print("=" * 50)
    
    init_database()
    print("✅ Database initialized successfully!")
    print()

def example_2_add_patient():
    """Add a new patient to the system"""
    print("=" * 50)
    print("Example 2: Add Patient")
    print("=" * 50)
    
    patient_data = {
        'Name': 'John Doe',
        'Age': 25,
        'Gender': 'Male',
        'Ethnicity': 'Caucasian',
        'Severity': 'Severe',
        'Mutation': 'Intron22',
        'Blood_Type': 'O',
        'HLA_Type': 'High Risk',
        'Dose': 50,
        'Exposure': 15,
        'Product_Type': 'Recombinant',
        'Treatment_Adherence': 85,
        'Family_History': 'Yes',
        'Previous_Inhibitor': 'No',
        'Joint_Damage': 25,
        'Bleeding_Episodes': 3,
        'Factor_Level': 2,
        'Immunosuppression': 'No',
        'Active_Infection': 'No',
        'Vaccination_Status': 'Complete',
        'Physical_Activity': 'Moderate',
        'Stress_Level': 'Moderate',
        'Comorbidities': 'None',
        'Risk_Score': 0.65
    }
    
    patient_id = add_patient(patient_data)
    print(f"✅ Patient added with ID: {patient_id}")
    print(f"   Name: {patient_data['Name']}")
    print(f"   Risk Score: {patient_data['Risk_Score']:.1%}")
    print()
    
    return patient_id

def example_3_retrieve_patient(patient_id):
    """Retrieve patient information"""
    print("=" * 50)
    print("Example 3: Retrieve Patient")
    print("=" * 50)
    
    patient = get_patient(patient_id)
    
    if patient:
        print(f"✅ Patient found:")
        print(f"   ID: {patient['id']}")
        print(f"   Name: {patient['name']}")
        print(f"   Severity: {patient['severity']}")
        print(f"   Mutation: {patient['mutation']}")
        print(f"   Risk Score: {patient['risk_score']:.1%}")
    else:
        print(f"❌ Patient {patient_id} not found")
    print()

def example_4_gpt_chatbot(patient_id):
    """Use GPT chatbot to get clinical recommendations"""
    print("=" * 50)
    print("Example 4: GPT Chatbot - Clinical Recommendations")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OpenAI API key not configured")
        print("   Set OPENAI_API_KEY in .env file")
        print()
        return
    
    patient = get_patient(patient_id)
    
    # Get recommendation
    try:
        print("🤖 Asking AI Doctor for clinical recommendations...")
        recommendations = get_clinical_recommendations(patient)
        print("\n📋 Clinical Recommendations:")
        print("-" * 50)
        print(recommendations[:500] + "...")  # Print first 500 chars
        print()
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
        print()

def example_5_add_conversation(patient_id):
    """Add a conversation to database"""
    print("=" * 50)
    print("Example 5: Add Conversation")
    print("=" * 50)
    
    user_msg = "What monitoring should this patient have?"
    
    if os.getenv("OPENAI_API_KEY"):
        patient = get_patient(patient_id)
        gpt_response = create_gpt_response(user_msg, patient_context=patient)
        
        add_conversation(patient_id, user_msg, gpt_response, "monitoring")
        print("✅ Conversation saved!")
        print(f"   User: {user_msg}")
        print(f"   AI Response: {gpt_response[:100]}...")
    else:
        print("⚠️  OpenAI API key not configured")
    print()

def example_6_add_doctor_note(patient_id):
    """Add a doctor note"""
    print("=" * 50)
    print("Example 6: Add Doctor Note")
    print("=" * 50)
    
    add_doctor_note(
        patient_id,
        doctor_name="Dr. Smith",
        note_content="Patient showing good compliance. No signs of inhibitor development. Continue current prophylaxis regimen.",
        note_category="Monitoring",
        severity="Normal"
    )
    
    print("✅ Doctor note added!")
    print("   Doctor: Dr. Smith")
    print("   Category: Monitoring")
    print()

def example_7_retrieve_doctor_notes(patient_id):
    """Retrieve doctor notes for patient"""
    print("=" * 50)
    print("Example 7: Retrieve Doctor Notes")
    print("=" * 50)
    
    notes = get_doctor_notes(patient_id)
    
    if notes:
        print(f"✅ Found {len(notes)} note(s):")
        for i, note in enumerate(notes, 1):
            print(f"\n   Note {i}:")
            print(f"   Doctor: {note['doctor_name']}")
            print(f"   Category: {note['note_category']}")
            print(f"   Severity: {note['severity']}")
            print(f"   Date: {note['created_at']}")
            print(f"   Content: {note['note_content'][:100]}...")
    else:
        print("ℹ️  No notes found")
    print()

def example_8_get_conversation_history(patient_id):
    """Retrieve conversation history"""
    print("=" * 50)
    print("Example 8: Get Conversation History")
    print("=" * 50)
    
    history = get_conversation_history(patient_id)
    
    if history:
        print(f"✅ Found {len(history)} conversation(s):")
        for i, msg in enumerate(history[:3], 1):  # Show first 3
            print(f"\n   Message {i}:")
            print(f"   User: {msg['user_message'][:80]}...")
            print(f"   AI: {msg['gpt_response'][:80]}...")
    else:
        print("ℹ️  No conversations yet")
    print()

def example_9_dashboard_stats():
    """Get dashboard statistics"""
    print("=" * 50)
    print("Example 9: Dashboard Statistics")
    print("=" * 50)
    
    stats = get_dashboard_stats()
    
    print("📊 System Statistics:")
    print(f"   Total Patients: {stats['total_patients']}")
    print(f"   High Risk: {stats['high_risk_patients']}")
    print(f"   Severe Cases: {stats['severe_cases']}")
    print(f"   Average Risk: {stats['average_risk']:.1%}")
    print(f"   Total Conversations: {stats['total_conversations']}")
    print(f"   Recent Notes: {stats['recent_notes']}")
    print()

def example_10_search_patients():
    """Search for patients"""
    print("=" * 50)
    print("Example 10: Search Patients")
    print("=" * 50)
    
    results = search_patients("Intron22")
    
    if results:
        print(f"✅ Found {len(results)} patient(s) with Intron22 mutation:")
        for patient in results[:3]:
            print(f"   - {patient['name']} (Risk: {patient['risk_score']:.1%})")
    else:
        print("ℹ️  No patients found")
    print()

def run_all_examples():
    """Run all examples in sequence"""
    print("\n")
    print("🏥" * 25)
    print("HEMOPHILIA AI PLATFORM - USAGE EXAMPLES")
    print("🏥" * 25)
    print("\n")
    
    # Initialize database
    example_1_initialize_database()
    
    # Add patient
    patient_id = example_2_add_patient()
    
    # Retrieve patient
    example_3_retrieve_patient(patient_id)
    
    # GPT recommendations (requires API key)
    example_4_gpt_chatbot(patient_id)
    
    # Add conversation (requires API key)
    example_5_add_conversation(patient_id)
    
    # Add doctor note
    example_6_add_doctor_note(patient_id)
    
    # Retrieve notes
    example_7_retrieve_doctor_notes(patient_id)
    
    # Get conversation history
    example_8_get_conversation_history(patient_id)
    
    # Dashboard stats
    example_9_dashboard_stats()
    
    # Search patients
    example_10_search_patients()
    
    print("=" * 50)
    print("✅ All examples completed!")
    print("=" * 50)

if __name__ == "__main__":
    run_all_examples()
