"""
Clinical Chatbot Configuration & Utilities
Easy-to-use module for integrating clinical AI chatbot into applications

Usage:
    from chatbot_config import ChatbotConfig, get_chatbot_instance
    
    config = ChatbotConfig()
    chatbot = get_chatbot_instance()
    response = chatbot.chat("Your clinical question")
"""

import os
import json
from typing import Dict, Optional
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

class ChatbotConfig:
    """Centralized configuration for clinical chatbot"""
    
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    OPENAI_MODELS = ["gpt-4", "gpt-3.5-turbo"]
    DEFAULT_MODEL = "gpt-4"
    
    # Database Configuration
    DATABASE_PATH = os.getenv("CHATBOT_DB", "clinical_conversations.db")
    DATABASE_TYPE = "sqlite"
    
    # Response Configuration
    TEMPERATURE_RISK = 0.5  # Lower for consistent risk assessment
    TEMPERATURE_RECOMMENDATION = 0.6  # Medium for balanced recommendations
    TEMPERATURE_CHAT = 0.7  # Higher for natural conversation
    
    MAX_TOKENS_RISK = 1500
    MAX_TOKENS_RECOMMENDATION = 2000
    MAX_TOKENS_CHAT = 1500
    MAX_TOKENS_MONITORING = 1500
    
    # System Prompts
    SYSTEM_PROMPT_CLINICAL = """You are an expert clinical decision support AI specialized in hemophilia care.
    You provide evidence-based clinical information to support healthcare professionals.
    Key principles:
    - Base recommendations on current clinical guidelines
    - Consider patient-specific factors
    - Highlight safety considerations
    - Always recommend professional clinical review
    - Use clear, professional medical language"""
    
    SYSTEM_PROMPT_RISK = """You are a clinical risk assessment expert.
    Provide detailed, evidence-based risk analysis based on patient data and clinical factors.
    Include pathophysiological mechanisms and clinical significance."""
    
    SYSTEM_PROMPT_TREATMENT = """You are an experienced clinical specialist in hemophilia management.
    Provide evidence-based treatment recommendations based on current guidelines and patient factors."""
    
    SYSTEM_PROMPT_MONITORING = """You are a clinical laboratory and data analyst.
    Analyze monitoring data and provide clinical interpretation with clear actionable recommendations."""
    
    # Session Configuration
    SESSION_TIMEOUT_MINUTES = 120
    MAX_CONVERSATION_LENGTH = 50
    
    # Feature Flags
    ENABLE_CONVERSATION_HISTORY = True
    ENABLE_CLINICAL_NOTES = True
    ENABLE_RISK_SCORING = True
    ENABLE_TREATMENT_RECOMMENDATIONS = True
    ENABLE_MONITORING_ANALYSIS = True
    
    # Logging
    LOG_CONVERSATIONS = True
    LOG_LEVEL = "INFO"


# ============================================================================
# INSTANCE MANAGEMENT
# ============================================================================

_chatbot_instance = None
_chatbot_config = None


def init_chatbot(db_path: str = None) -> None:
    """
    Initialize chatbot instance
    
    Args:
        db_path: Optional database path override
    """
    global _chatbot_instance, _chatbot_config
    
    from clinical_ai_chatbot import ClinicalAIChatbot
    
    _chatbot_config = ChatbotConfig()
    
    if db_path:
        _chatbot_config.DATABASE_PATH = db_path
    
    _chatbot_instance = ClinicalAIChatbot(_chatbot_config.DATABASE_PATH)
    
    print("✅ Clinical AI Chatbot initialized successfully")


def get_chatbot_instance() -> Optional['ClinicalAIChatbot']:
    """
    Get chatbot instance (lazy initialization)
    
    Returns:
        ClinicalAIChatbot instance
    """
    global _chatbot_instance
    
    if _chatbot_instance is None:
        init_chatbot()
    
    return _chatbot_instance


def get_config() -> ChatbotConfig:
    """
    Get configuration instance
    
    Returns:
        ChatbotConfig instance
    """
    global _chatbot_config
    
    if _chatbot_config is None:
        _chatbot_config = ChatbotConfig()
    
    return _chatbot_config


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_configuration() -> Dict[str, bool]:
    """
    Validate chatbot configuration
    
    Returns:
        Validation results dict
    """
    config = get_config()
    results = {}
    
    # Check API key
    results["api_key_configured"] = bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your-api-key-here")
    
    # Check database path
    results["database_path_valid"] = bool(config.DATABASE_PATH)
    
    # Check database exists or can be created
    try:
        db_path = Path(config.DATABASE_PATH)
        results["database_accessible"] = db_path.parent.exists()
    except:
        results["database_accessible"] = False
    
    # Check required packages
    packages = {
        "openai": "openai",
        "sqlite3": "sqlite3"
    }
    
    for name, module_name in packages.items():
        try:
            __import__(module_name)
            results[f"package_{name}_installed"] = True
        except ImportError:
            results[f"package_{name}_installed"] = False
    
    return results


def print_configuration_status() -> None:
    """Print configuration status to console"""
    
    config = get_config()
    validation = validate_configuration()
    
    print("\n" + "="*60)
    print("🏥 CLINICAL AI CHATBOT - CONFIGURATION STATUS")
    print("="*60 + "\n")
    
    print("📋 Configuration Settings:")
    print(f"  - API Key Configured: {'✅' if validation['api_key_configured'] else '❌'}")
    print(f"  - Database Path: {config.DATABASE_PATH}")
    print(f"  - Default Model: {config.DEFAULT_MODEL}")
    print(f"  - Session Timeout: {config.SESSION_TIMEOUT_MINUTES} minutes")
    
    print("\n📦 Package Status:")
    for key, value in validation.items():
        if key.startswith("package_"):
            status = "✅" if value else "❌"
            package = key.replace("package_", "").replace("_", " ").title()
            print(f"  - {package}: {status}")
    
    print("\n🔧 Feature Flags:")
    print(f"  - Conversation History: {'✅' if config.ENABLE_CONVERSATION_HISTORY else '❌'}")
    print(f"  - Clinical Notes: {'✅' if config.ENABLE_CLINICAL_NOTES else '❌'}")
    print(f"  - Risk Scoring: {'✅' if config.ENABLE_RISK_SCORING else '❌'}")
    print(f"  - Treatment Recommendations: {'✅' if config.ENABLE_TREATMENT_RECOMMENDATIONS else '❌'}")
    print(f"  - Monitoring Analysis: {'✅' if config.ENABLE_MONITORING_ANALYSIS else '❌'}")
    
    print("\n✅ Ready to use!" if all(validation.values()) else "\n⚠️ Some issues detected - see above")
    print("="*60 + "\n")


# ============================================================================
# QUICK USE FUNCTIONS
# ============================================================================

def quick_risk_analysis(
    patient_data: Dict,
    assessment_data: Dict
) -> str:
    """
    Quick risk analysis without full session setup
    
    Args:
        patient_data: Patient information dict
        assessment_data: Risk assessment data dict
        
    Returns:
        Risk explanation text
    """
    from clinical_ai_chatbot import PatientContext
    
    chatbot = get_chatbot_instance()
    
    # Create patient context
    patient = PatientContext(
        patient_id=patient_data.get("patient_id", "TEMP"),
        age=patient_data.get("age", 30),
        gender=patient_data.get("gender", "M"),
        diagnosis=patient_data.get("diagnosis", "Hemophilia"),
        severity=patient_data.get("severity", "moderate"),
        treatment_history=patient_data.get("treatment_history", []),
        current_medications=patient_data.get("current_medications", []),
        allergies=patient_data.get("allergies", []),
        recent_labs=patient_data.get("recent_labs")
    )
    
    # Start session and analyze
    chatbot.start_session(patient)
    risk_explanation = chatbot.generate_risk_explanation(assessment_data)
    chatbot.end_session()
    
    return risk_explanation


def quick_recommendations(
    patient_data: Dict,
    clinical_question: str
) -> str:
    """
    Quick clinical recommendations without full session setup
    
    Args:
        patient_data: Patient information dict
        clinical_question: Clinical question or topic
        
    Returns:
        Recommendations text
    """
    from clinical_ai_chatbot import PatientContext
    
    chatbot = get_chatbot_instance()
    
    # Create patient context
    patient = PatientContext(
        patient_id=patient_data.get("patient_id", "TEMP"),
        age=patient_data.get("age", 30),
        gender=patient_data.get("gender", "M"),
        diagnosis=patient_data.get("diagnosis", "Hemophilia"),
        severity=patient_data.get("severity", "moderate"),
        treatment_history=patient_data.get("treatment_history", []),
        current_medications=patient_data.get("current_medications", []),
        allergies=patient_data.get("allergies", [])
    )
    
    # Start session and get recommendations
    chatbot.start_session(patient)
    recommendations = chatbot.get_clinical_recommendations(clinical_question)
    chatbot.end_session()
    
    return recommendations


def quick_monitoring_analysis(
    patient_data: Dict,
    monitoring_data: Dict
) -> str:
    """
    Quick monitoring data analysis
    
    Args:
        patient_data: Patient information dict
        monitoring_data: Monitoring data dict
        
    Returns:
        Analysis text
    """
    from clinical_ai_chatbot import PatientContext
    
    chatbot = get_chatbot_instance()
    
    # Create patient context
    patient = PatientContext(
        patient_id=patient_data.get("patient_id", "TEMP"),
        age=patient_data.get("age", 30),
        gender=patient_data.get("gender", "M"),
        diagnosis=patient_data.get("diagnosis", "Hemophilia"),
        severity=patient_data.get("severity", "moderate"),
        treatment_history=patient_data.get("treatment_history", []),
        current_medications=patient_data.get("current_medications", []),
        allergies=patient_data.get("allergies", [])
    )
    
    # Start session and analyze
    chatbot.start_session(patient)
    analysis = chatbot.analyze_monitoring_data(monitoring_data)
    chatbot.end_session()
    
    return analysis


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """
    Setup environment variables for chatbot
    
    Creates .env file if not exists with template
    """
    env_path = Path(".env")
    
    if not env_path.exists():
        env_content = """# Clinical AI Chatbot Configuration
OPENAI_API_KEY=your-api-key-here
CHATBOT_DB=clinical_conversations.db
LOG_LEVEL=INFO
DEBUG_MODE=False
"""
        env_path.write_text(env_content)
        print(f"✅ Created .env template at {env_path}")
        print("   Please add your OpenAI API key to continue")
    else:
        print(f"✅ .env file exists at {env_path}")


def load_environment():
    """Load environment variables from .env file"""
    from pathlib import Path
    
    env_path = Path(".env")
    
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Environment variables loaded from .env")


# ============================================================================
# EXPORT HELPERS
# ============================================================================

def export_conversation(conversation_id: int, format: str = "json") -> str:
    """
    Export conversation in specified format
    
    Args:
        conversation_id: Conversation ID
        format: Export format (json, csv, txt)
        
    Returns:
        Exported data as string
    """
    from clinical_chatbot_integration import ConversationExporter
    
    exporter = ConversationExporter()
    
    if format == "json":
        return exporter.export_to_json(conversation_id)
    elif format == "csv":
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            exporter.export_to_csv(conversation_id, f.name)
            return Path(f.name).read_text()
    elif format == "txt":
        return exporter.export_text_summary(conversation_id)
    else:
        raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo():
    """Run configuration demo"""
    
    print("\n" + "="*70)
    print("🏥 CLINICAL AI CHATBOT - CONFIGURATION DEMO")
    print("="*70 + "\n")
    
    # 1. Setup environment
    print("1️⃣ Setting up environment...")
    setup_environment()
    load_environment()
    
    # 2. Check configuration
    print("\n2️⃣ Checking configuration...")
    print_configuration_status()
    
    # 3. Get instances
    print("3️⃣ Getting chatbot instance...")
    chatbot = get_chatbot_instance()
    config = get_config()
    print(f"✅ Chatbot ready - Using model: {config.DEFAULT_MODEL}")
    
    # 4. Quick examples
    print("\n4️⃣ Quick helper function examples:\n")
    
    patient_example = {
        "patient_id": "PAT-2026-001",
        "age": 35,
        "gender": "M",
        "diagnosis": "Hemophilia A",
        "severity": "moderate",
        "treatment_history": ["Factor VIII replacement"],
        "current_medications": ["Factor VIII"],
        "allergies": []
    }
    
    print("   Example patient data:")
    print(f"   {json.dumps(patient_example, indent=6)}\n")
    
    print("   Available quick functions:")
    print("   - quick_risk_analysis(patient_data, assessment_data)")
    print("   - quick_recommendations(patient_data, clinical_question)")
    print("   - quick_monitoring_analysis(patient_data, monitoring_data)\n")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    demo()
