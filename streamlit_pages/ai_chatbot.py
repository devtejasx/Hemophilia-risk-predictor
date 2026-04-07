"""
AI Chatbot Page - Interactive conversational interface
"""

import streamlit as st
from datetime import datetime
from streamlit_utils.state_manager import StateManager
from streamlit_utils.ui_components import (
    create_header, show_info, show_success, create_tabs
)
from streamlit_utils.backend_client import get_backend_client


def render():
    """Render AI chatbot page"""
    
    create_header("💬 AI Clinical Assistant", "Ask questions about patient data and clinical insights")
    
    # Get clients
    state = StateManager()
    backend = get_backend_client()
    
    # Get current patient
    current_patient = state.get_current_patient()
    
    # Information section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👤 Patient", current_patient.get("patient_id", "None") if current_patient else "None")
    with col2:
        st.metric("🤖 Model", "GPT-4")
    with col3:
        st.metric("📡 Status", "🟢 Online" if backend.health_check() else "🔴 Offline")
    
    st.divider()
    
    # ========================================================================
    # CHAT INTERFACE
    # ========================================================================
    
    tab1, tab2 = st.tabs(["Chat", "Examples"])
    
    with tab1:
        st.markdown("### 💬 Chat History")
        
        # Get or initialize chat history
        chat_history = state.get_chat_history()
        
        # Display chat messages
        chat_container = st.container()
        
        with chat_container:
            for message in chat_history:
                role = message.get("role", "assistant")
                content = message.get("content", "")
                timestamp = message.get("timestamp", "")
                
                if role == "user":
                    st.markdown(f"""
                    <div style='
                        background-color: rgba(0, 212, 255, 0.1);
                        padding: 1rem;
                        border-radius: 8px;
                        margin: 0.5rem 0;
                        border-left: 4px solid #00d4ff;
                    '>
                        <p style='font-size: 0.8em; color: #888; margin: 0;'>{timestamp}</p>
                        <p style='margin: 0.5rem 0 0 0;'>👤 <strong>You:</strong></p>
                        <p style='margin: 0.5rem 0 0 0;'>{content}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='
                        background-color: rgba(0, 150, 180, 0.1);
                        padding: 1rem;
                        border-radius: 8px;
                        margin: 0.5rem 0;
                        border-left: 4px solid #0099cc;
                    '>
                        <p style='font-size: 0.8em; color: #888; margin: 0;'>{timestamp}</p>
                        <p style='margin: 0.5rem 0 0 0;'>🤖 <strong>Assistant:</strong></p>
                        <p style='margin: 0.5rem 0 0 0;'>{content}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # Input section
        st.markdown("### 📝 Send Message")
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your message (what would you like to know?)",
                placeholder="e.g., What is this patient's current risk level?",
                key="chat_input"
            )
        
        with col2:
            send_button = st.button("📤 Send", use_container_width=True)
        
        # Process message
        if send_button and user_input.strip():
            # Add user message to history
            state.add_chat_message("user", user_input)
            
            # Get backend context
            patient_id = current_patient.get("patient_id") if current_patient else None
            
            # Send to chatbot
            with st.spinner("🤖 AI is thinking..."):
                response = backend.send_chat_message(user_input, patient_id)
            
            if response:
                # Add assistant response to history
                state.add_chat_message("assistant", response)
                show_success("✅ Response received!")
                st.rerun()
            else:
                st.error("❌ Failed to get response. Check backend connection.")
        
        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            state.clear_chat_history()
            st.success("Chat history cleared")
            st.rerun()
    
    with tab2:
        st.markdown("### 💡 Example Questions")
        
        st.markdown("#### Patient Information")
        examples_info = [
            "What is the patient's current medical status?",
            "Who is the current patient?",
            "What are the patient's vital signs?",
            "What diagnosis does this patient have?",
        ]
        
        for example in examples_info:
            if st.button(f"💬 {example}", use_container_width=True):
                state.add_chat_message("user", example)
                with st.spinner("🤖 AI is thinking..."):
                    response = backend.send_chat_message(example, current_patient.get("patient_id") if current_patient else None)
                if response:
                    state.add_chat_message("assistant", response)
                    show_success("Response received!")
                    st.rerun()
        
        st.divider()
        
        st.markdown("#### Clinical Analysis")
        examples_clinical = [
            "What are the main risk factors for this patient?",
            "What is the predicted risk level?",
            "Should this patient be monitored closely?",
            "What clinical interventions are recommended?",
        ]
        
        for example in examples_clinical:
            if st.button(f"⚕️ {example}", use_container_width=True):
                state.add_chat_message("user", example)
                with st.spinner("🤖 AI is thinking..."):
                    response = backend.send_chat_message(example, current_patient.get("patient_id") if current_patient else None)
                if response:
                    state.add_chat_message("assistant", response)
                    show_success("Response received!")
                    st.rerun()
        
        st.divider()
        
        st.markdown("#### Historical Analysis")
        examples_history = [
            "What trends do you see in the patient's history?",
            "Has the patient's condition improved or worsened?",
            "What was the patient's risk score in the past?",
            "When was the highest risk recorded?",
        ]
        
        for example in examples_history:
            if st.button(f"📊 {example}", use_container_width=True):
                state.add_chat_message("user", example)
                with st.spinner("🤖 AI is thinking..."):
                    response = backend.send_chat_message(example, current_patient.get("patient_id") if current_patient else None)
                if response:
                    state.add_chat_message("assistant", response)
                    show_success("Response received!")
                    st.rerun()
    
    # ========================================================================
    # INFO BOX
    # ========================================================================
    
    st.divider()
    st.markdown("### ℹ️ About This Assistant")
    
    st.markdown("""
    **Medical AI Clinical Assistant** - Powered by GPT-4
    
    This conversational interface provides:
    - 📊 Patient data analysis and interpretation
    - 🔍 Risk factor explanations
    - 💊 Clinical recommendations
    - 📈 Trend analysis and insights
    - 🎯 Risk stratification
    
    **Important**: This AI provides clinical decision support only and should not replace professional medical judgment.
    All recommendations should be reviewed by qualified healthcare providers.
    
    **Features**:
    - Context-aware responses using current patient data
    - Integration with ML predictions
    - Plain language explanations
    - Evidence-based recommendations
    """)
    
    # Disclaimer
    st.warning("""
    ⚠️ **DISCLAIMER**: This is an AI assistant for clinical decision support.
    Always consult qualified healthcare professionals before making clinical decisions.
    """)
