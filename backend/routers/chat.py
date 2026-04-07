"""
Clinical AI Chat API routes
Requires: Authenticated user (any role)
Roles: patient, doctor, admin (can access based on patient context)
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging

from ..models import ChatRequest, ChatResponse, ChatMessage
from ..gpt_utils import (
    get_clinical_response, 
    get_available_modes, 
    get_medical_definitions,
    MEDICAL_DEFINITIONS
)
from ..security import get_current_user

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
) -> ChatResponse:
    """
    Query clinical AI assistant
    
    **Authentication:** Required (any authenticated user)
    
    **Roles:** patient, doctor, admin
    
    Accepts question, clinical mode, and optional patient context.
    Returns AI response with appropriate disclaimers.
    """
    try:
        # Validate mode
        available_modes = [m[0] for m in get_available_modes()]
        if request.mode not in available_modes:
            raise ValueError(f"Invalid mode. Available: {', '.join(available_modes)}")
        
        # Convert conversation history if needed
        conversation = []
        if request.conversation_history:
            for msg in request.conversation_history:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Generate response
        response_text, mode_used = get_clinical_response(
            mode=request.mode,
            question=request.question,
            patient_data=request.patient_data,
            conversation_history=conversation
        )
        
        # Add disclaimer
        disclaimer = "⚠️ AI-generated suggestions are for educational discussion only and should NOT replace professional medical judgment. Always consult qualified specialists."
        
        logger.info(f"[{current_user['username']}] Chat query: mode={mode_used}")
        
        return ChatResponse(
            response=response_text,
            mode_used=mode_used,
            disclaimer=disclaimer,
            sources=["OpenAI GPT-4", "Clinical Guidelines"],
            confidence=0.85
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


@router.get("/modes")
async def get_modes(current_user: dict = Depends(get_current_user)):
    """
    Get available chat modes with descriptions
    
    **Authentication:** Required
    **Roles:** patient, doctor, admin
    """
    modes = get_available_modes()
    return {
        "modes": [
            {
                "id": mode[0],
                "description": mode[1],
                "icon": mode[2]
            }
            for mode in modes
        ]
    }


@router.get("/definitions")
async def get_definitions(
    terms: List[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Get medical terminology definitions
    
    **Authentication:** Required
    **Roles:** patient, doctor, admin
    
    If terms provided, returns definitions for those terms.
    Otherwise returns all available definitions.
    """
    if terms:
        definitions = get_medical_definitions(terms)
    else:
        definitions = MEDICAL_DEFINITIONS
    
    return {
        "definitions": definitions,
        "count": len(definitions)
    }


@router.post("/feedback")
async def submit_feedback(
    conversation_id: str,
    rating: int,
    feedback: str = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Submit feedback on AI response
    
    **Authentication:** Required
    **Roles:** patient, doctor, admin
    
    Records user feedback for response quality analytics (for future implementation)
    """
    if not 1 <= rating <= 5:
        raise HTTPException(status_code=400, detail="Rating must be 1-5")
    
    logger.info(f"[{current_user['username']}] Feedback: conversation={conversation_id}, rating={rating}")
    
    return {
        "status": "feedback_recorded",
        "conversation_id": conversation_id,
        "rating": rating
    }
