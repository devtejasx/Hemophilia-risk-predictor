"""
Chat Router
Endpoints for AI chatbot
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
from models import ChatMessage, ChatResponse
from services.chat_service import chat_service
from database import get_db
from exceptions import ChatException, exception_to_http_exception

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def send_message(
    chat_message: ChatMessage,
    db = Depends(get_db)
):
    """
    Send message to clinical AI chatbot
    
    Args:
        chat_message: User message with optional patient context
        db: Database connection
        
    Returns:
        AI clinical response
    """
    try:
        # Get patient context if available
        patient_context = None
        if chat_message.patient_id:
            try:
                query = 'SELECT * FROM patients WHERE id = ?'
                cursor = db.cursor()
                cursor.execute(query, (chat_message.patient_id,))
                row = cursor.fetchone()
                
                if row:
                    patient_context = dict(row)
            except Exception as e:
                print(f"Warning: Failed to fetch patient context: {e}")
        
        # Get AI response
        response_text = chat_service.get_clinical_response(
            chat_message.message,
            patient_context,
            chat_message.mode or "general"
        )
        
        # Save conversation to database
        if chat_message.patient_id and db:
            try:
                query = '''
                    INSERT INTO conversations 
                    (patient_id, user_message, ai_response)
                    VALUES (?, ?, ?)
                '''
                cursor = db.cursor()
                cursor.execute(query, (
                    chat_message.patient_id,
                    chat_message.message,
                    response_text
                ))
            except Exception as e:
                print(f"Warning: Failed to save conversation: {e}")
        
        return ChatResponse(
            message_id=str(uuid.uuid4()),
            response=response_text,
            confidence=0.85,
            timestamp=datetime.utcnow()
        )
        
    except ChatException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/patient/{patient_id}")
async def get_conversation_history(
    patient_id: int,
    limit: int = 50,
    db = Depends(get_db)
):
    """
    Get conversation history for patient
    
    Args:
        patient_id: Patient ID
        limit: Maximum number of messages
        db: Database connection
        
    Returns:
        Conversation history
    """
    try:
        query = '''
            SELECT * FROM conversations 
            WHERE patient_id = ? 
            ORDER BY created_at DESC
            LIMIT ?
        '''
        cursor = db.cursor()
        cursor.execute(query, (patient_id, limit))
        conversations = cursor.fetchall()
        
        return {
            "patient_id": patient_id,
            "messages": [dict(row) for row in conversations],
            "total": len(conversations)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health")
async def check_chat_service():
    """Check if chat service is available"""
    return {
        "service": "Chat Service",
        "available": chat_service.is_available(),
        "model": chat_service.model if chat_service.is_available() else None
    }
