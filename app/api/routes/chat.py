from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from app.models.chat import ChatRequest, ChatResponse, ChatMessage
from app.services.openai_service import OpenAIService
from app.services.memory_service import MemoryService
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory conversation storage (in production, use a proper database)
conversation_history: Dict[str, List[Dict[str, str]]] = {}

def get_openai_service():
    return OpenAIService()

def get_memory_service():
    try:
        return MemoryService()
    except Exception as e:
        logger.error(f"Failed to initialize memory service: {e}")
        # Return a minimal memory service that doesn't use Pinecone
        return MemoryService()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    openai_service: OpenAIService = Depends(get_openai_service),
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Process a chat message with optional memory retrieval."""
    try:
        session_id = request.session_id
        
        # Initialize conversation history if not exists
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Add user message to history
        user_message = {"role": "user", "content": request.message}
        conversation_history[session_id].append(user_message)
        
        # AI decides whether to use memory
        use_memory = request.use_memory and openai_service.should_use_memory(
            request.message, 
            conversation_history[session_id]
        )
        
        memories = []
        if use_memory:
            try:
                # Retrieve relevant memories
                memories = memory_service.retrieve_memories(
                    query=request.message,
                    session_id=session_id,
                    user_id=request.user_id
                )
                logger.info(f"Retrieved {len(memories)} memories for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")
                memories = []
        
        # Generate AI response
        response_data = openai_service.chat_completion(
            messages=conversation_history[session_id],
            use_memory=use_memory,
            memories=memories
        )
        
        # Add AI response to history
        ai_message = {"role": "assistant", "content": response_data["response"]}
        conversation_history[session_id].append(ai_message)
        
        # Store the conversation in memory
        try:
            memory_service.store_memory(
                content=f"User: {request.message}\nAssistant: {response_data['response']}",
                session_id=session_id,
                user_id=request.user_id,
                metadata={
                    "message_type": "conversation",
                    "context": request.context
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")
        
        return ChatResponse(
            response=response_data["response"],
            session_id=session_id,
            memory_used=use_memory,
            memories_retrieved=memories,
            tokens_used=response_data.get("tokens_used"),
            model_used=response_data.get("model_used")
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session."""
    try:
        if session_id not in conversation_history:
            return {"messages": [], "session_id": session_id}
        
        return {
            "messages": conversation_history[session_id],
            "session_id": session_id,
            "message_count": len(conversation_history[session_id])
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/conversation/{session_id}")
async def clear_conversation(
    session_id: str,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Clear conversation history and memories for a session."""
    try:
        # Clear in-memory history
        if session_id in conversation_history:
            del conversation_history[session_id]
        
        # Clear memories from vector store
        try:
            deleted_count = memory_service.delete_session_memories(session_id)
        except Exception as e:
            logger.warning(f"Failed to delete memories: {e}")
            deleted_count = 0
        
        return {
            "message": f"Conversation cleared for session {session_id}",
            "memories_deleted": deleted_count,
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session")
async def create_session(user_id: str = None):
    """Create a new conversation session."""
    try:
        session_id = str(uuid.uuid4())
        conversation_history[session_id] = []
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def list_sessions():
    """List all active conversation sessions."""
    try:
        sessions = []
        for session_id in conversation_history.keys():
            sessions.append({
                "session_id": session_id,
                "message_count": len(conversation_history[session_id]),
                "last_message": conversation_history[session_id][-1]["content"][:100] if conversation_history[session_id] else None
            })
        
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 