from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: Optional[str] = None
    use_memory: bool = True
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memory_used: bool = False
    memories_retrieved: List[Dict[str, Any]] = []
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None

class MemoryItem(BaseModel):
    id: str
    content: str
    session_id: str
    user_id: Optional[str] = None
    timestamp: datetime
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    similarity_score: Optional[float] = None

class MemoryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 5
    threshold: float = 0.7

class MemoryResponse(BaseModel):
    memories: List[MemoryItem]
    query: str
    total_found: int
    threshold_used: float

class ConversationSession(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    metadata: Optional[Dict[str, Any]] = None 