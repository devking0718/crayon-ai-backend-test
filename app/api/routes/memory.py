from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from app.models.chat import MemoryRequest, MemoryResponse, MemoryItem
from app.services.memory_service import MemoryService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

def get_memory_service():
    return MemoryService()

@router.post("/memory/search", response_model=MemoryResponse)
async def search_memories(
    request: MemoryRequest,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Search for relevant memories based on a query."""
    try:
        memories = memory_service.retrieve_memories(
            query=request.query,
            session_id=request.session_id,
            user_id=request.user_id,
            limit=request.limit,
            threshold=request.threshold
        )
        
        return MemoryResponse(
            memories=memories,
            query=request.query,
            total_found=len(memories),
            threshold_used=request.threshold
        )
        
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/store")
async def store_memory(
    content: str,
    session_id: str,
    user_id: str = None,
    metadata: Dict[str, Any] = None,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Store a new memory in the vector database."""
    try:
        memory_id = memory_service.store_memory(
            content=content,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        
        return {
            "memory_id": memory_id,
            "message": "Memory stored successfully",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/memory/{memory_id}")
async def delete_memory(
    memory_id: str,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Delete a specific memory from the vector database."""
    try:
        success = memory_service.delete_memory(memory_id)
        
        if success:
            return {"message": f"Memory {memory_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Memory not found")
            
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/memory/session/{session_id}")
async def delete_session_memories(
    session_id: str,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Delete all memories for a specific session."""
    try:
        deleted_count = memory_service.delete_session_memories(session_id)
        
        return {
            "message": f"Deleted {deleted_count} memories for session {session_id}",
            "deleted_count": deleted_count,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error deleting session memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory/stats")
async def get_memory_stats(
    session_id: str = None,
    user_id: str = None,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Get statistics about stored memories."""
    try:
        stats = memory_service.get_memory_stats(session_id, user_id)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory/session/{session_id}")
async def get_session_memories(
    session_id: str,
    limit: int = 50,
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Get all memories for a specific session."""
    try:
        # Use a generic query to get all memories for the session
        memories = memory_service.retrieve_memories(
            query="",  # Empty query to get all memories
            session_id=session_id,
            limit=limit,
            threshold=0.0  # No threshold to get all memories
        )
        
        return {
            "memories": memories,
            "session_id": session_id,
            "total": len(memories)
        }
        
    except Exception as e:
        logger.error(f"Error getting session memories: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 