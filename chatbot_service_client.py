"""
Chatbot Service Client for Streamlit
Connects to Express backend service with MongoDB persistence
"""

import requests
import json
import streamlit as st
from typing import Dict, Optional, Tuple
from datetime import datetime

class ChatbotServiceClient:
    """Client for communicating with chatbot service backend"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        """
        Initialize chatbot client
        
        Args:
            base_url: Base URL of chatbot service (default: local)
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        self.session_id = None
        self.is_connected = False
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check if backend service is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            self.is_connected = response.status_code == 200
            return self.is_connected
        except requests.exceptions.RequestException:
            self.is_connected = False
            return False
    
    def start_conversation(self) -> bool:
        """
        Start new conversation session
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected:
            if not self._check_connection():
                st.error("⚠️ Chatbot service is not available. Is the server running on port 5001?")
                return False
        
        try:
            response = requests.get(f"{self.api_url}/start", timeout=5)
            response.raise_for_status()
            
            data = response.json()
            self.session_id = data.get('sessionId')
            
            if self.session_id:
                st.session_state.chatbot_session_id = self.session_id
                st.session_state.chatbot_greeting = data.get('greeting', '')
                st.session_state.chatbot_messages = [
                    {"role": "assistant", "content": data.get('greeting', '')}
                ]
                return True
        except Exception as e:
            st.error(f"❌ Error starting conversation: {str(e)}")
        
        return False
    
    def send_message(self, message: str) -> Optional[Dict]:
        """
        Send message to chatbot and get response
        
        Args:
            message: User message
            
        Returns:
            dict: Response data or None if failed
        """
        if not self.is_connected:
            if not self._check_connection():
                st.error("⚠️ Service unavailable. Check if server is running.")
                return None
        
        if not self.session_id or not st.session_state.get('chatbot_session_id'):
            if not self.start_conversation():
                return None
        
        try:
            response = requests.post(
                f"{self.api_url}/message",
                json={
                    "sessionId": st.session_state.get('chatbot_session_id'),
                    "message": message
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            st.error(f"❌ Error sending message: {str(e)}")
            return None
    
    def get_history(self) -> Optional[Dict]:
        """
        Get conversation history
        
        Returns:
            dict: Conversation data or None if failed
        """
        if not st.session_state.get('chatbot_session_id'):
            return None
        
        try:
            response = requests.get(
                f"{self.api_url}/history/{st.session_state.get('chatbot_session_id')}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            st.warning(f"Could not retrieve history: {str(e)}")
            return None
    
    def clear_conversation(self) -> bool:
        """
        Clear current conversation
        
        Returns:
            bool: True if successful
        """
        if not st.session_state.get('chatbot_session_id'):
            return False
        
        try:
            response = requests.post(
                f"{self.api_url}/clear/{st.session_state.get('chatbot_session_id')}",
                timeout=5
            )
            response.raise_for_status()
            
            # Reset local session state
            st.session_state.chatbot_messages = [
                {"role": "assistant", "content": "Conversation cleared. How can I help?"}
            ]
            return True
        
        except Exception as e:
            st.error(f"Error clearing conversation: {str(e)}")
            return False
    
    def get_service_status(self) -> Dict:
        """Get service status"""
        status = {
            "connected": self.is_connected,
            "service_url": self.base_url,
            "session_active": bool(st.session_state.get('chatbot_session_id'))
        }
        return status


def display_chat_interface(client: ChatbotServiceClient, container=None):
    """
    Display chat interface in Streamlit
    
    Args:
        client: ChatbotServiceClient instance
        container: Streamlit container to display in (optional)
    """
    
    # Initialize session state
    if 'chatbot_session_id' not in st.session_state:
        st.session_state.chatbot_messages = []
        if not client.start_conversation():
            st.error("❌ Failed to start chatbot service. Please ensure the service is running.")
            return
    
    # Display chat container
    display_target = container if container else st
    
    # Messages display area
    messages_container = display_target.container()
    with messages_container:
        for message in st.session_state.chatbot_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Input area
    col1, col2 = display_target.columns([1, 0.1])
    
    with col1:
        user_input = st.chat_input("Ask about hemophilia...", key="chat_input")
    
    with col2:
        if st.button("🔄", help="Clear conversation"):
            if client.clear_conversation():
                st.rerun()
    
    # Process user input
    if user_input:
        # Add user message to display
        st.session_state.chatbot_messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Get response from service
        with st.spinner("Thinking..."):
            response = client.send_message(user_input)
        
        if response and response.get('success'):
            # Add assistant response
            st.session_state.chatbot_messages.append({
                "role": "assistant",
                "content": response.get('response', 'No response')
            })
            
            # Show metadata
            with st.expander("ℹ️ Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Category", response.get('category', 'unknown'))
                with col2:
                    st.metric("Found Match", "✅" if response.get('found') else "❌")
                with col3:
                    st.metric("Messages", response.get('messageCount', 0))
            
            st.rerun()
        else:
            st.error("Failed to get response from service")


def init_chatbot_service(backend_url: str = "http://localhost:5001") -> ChatbotServiceClient:
    """
    Initialize chatbot service client
    
    Args:
        backend_url: URL of chatbot service backend
        
    Returns:
        ChatbotServiceClient: Initialized client
    """
    client = ChatbotServiceClient(backend_url)
    
    if not client.is_connected:
        st.warning(f"⚠️ Cannot connect to chatbot service at {backend_url}")
        st.info("To use the chatbot, please start the service:")
        st.code("cd chatbot-service\nnpm install\nnpm start", language="bash")
    
    return client
