"""
Chatbot Page - AI Assistant interface
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.navbar import show_page_header
from components.cards import info_card
from utils.session_state import add_chat_message, get_session_var, clear_chat_history
from services.chatbot_service import get_chatbot_service

st.set_page_config(page_title="Chatbot", layout="wide")


def main():
    show_page_header("🤖 AI Clinical Assistant", "Chat with our AI for clinical insights")
    
    # Info about chatbot
    info_card(
        "Chat Capabilities",
        "Ask about patient risk assessment, treatment recommendations, clinical questions, and more. "
        "This assistant provides contextual information to support clinical decision-making.",
        icon="💬"
    )
    
    st.divider()
    
    # Initialize chatbot service
    chatbot = get_chatbot_service()
    
    # Chat history container
    st.markdown("### Conversation")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.get('chat_history', []):
            with st.chat_message(message['role']):
                st.markdown(message['content'])
    
    st.divider()
    
    # Chat input
    col1, col2 = st.columns([20, 1])
    
    with col1:
        user_input = st.chat_input("Type your question or message...")
    
    with col2:
        clear_btn = st.button("🗑️", help="Clear chat history")
    
    if clear_btn:
        clear_chat_history()
        st.rerun()
    
    # Process user input
    if user_input:
        # Add user message
        add_chat_message("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get bot response
        bot_response = chatbot.get_response(user_input)
        add_chat_message("assistant", bot_response)
        
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        
        st.rerun()
    
    st.divider()
    
    # Quick commands
    st.markdown("### Quick Commands")
    col1, col2, col3, col4 = st.columns(4)
    
    commands = [
        ("Hello", "👋 Start conversation"),
        ("Help", "📚 Get help"),
        ("Risk", "⚠️ Discuss risk"),
        ("Treatment", "💊 Treatment info"),
    ]
    
    for i, (cmd, desc) in enumerate(commands):
        with [col1, col2, col3, col4][i]:
            if st.button(f"{cmd}\n{desc}", use_container_width=True):
                add_chat_message("user", cmd)
                bot_response = chatbot.get_response(cmd)
                add_chat_message("assistant", bot_response)
                st.rerun()
    
    st.divider()
    
    # Chat info
    st.markdown("""
    ### How to Use
    
    1. **Type your question** in the chat box below
    2. **Press Enter** to send
    3. **View responses** from the AI assistant
    4. **Use quick commands** for common questions
    5. **Clear history** to start a new conversation
    
    ### Example Questions
    - "What factors affect hemophilia risk?"
    - "How should treatment adherence be monitored?"
    - "What are the severity classifications?"
    - "How are risk scores calculated?"
    """)


if __name__ == "__main__":
    main()
