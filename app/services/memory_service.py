import pinecone
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services.openai_service import OpenAIService
import uuid
from datetime import datetime
import logging
import time
import requests

logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(self):
        self.openai_service = OpenAIService()
        self.index_name = settings.pinecone_index_name
        self.threshold = settings.memory_similarity_threshold
        self.max_retrieve = settings.max_memories_to_retrieve
        
        # Check if Pinecone is configured
        if not settings.pinecone_api_key:
            logger.warning("Pinecone API key not configured. Memory service will use in-memory storage.")
            self.pinecone_available = False
            self.index = None
            self._in_memory_storage = {}
            return
        
        # Initialize Pinecone with retry logic
        self.pinecone_available = False
        self.index = None
        self._in_memory_storage = {}
        
        # Try to initialize Pinecone with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to initialize Pinecone (attempt {attempt + 1}/{max_retries})")
                
                # Initialize Pinecone with new API
                self.pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
                
                # Test the connection by listing indexes
                available_indexes = self.pc.list_indexes()
                logger.info(f"Successfully connected to Pinecone. Available indexes: {available_indexes}")
                
                # Get or create index
                if self.index_name not in [index.name for index in available_indexes]:
                    logger.info(f"Creating Pinecone index: {self.index_name}")
                    self._create_index()
                    # Wait a moment for index to be ready
                    time.sleep(2)
                else:
                    # Check if existing index has correct dimension
                    logger.info(f"Pinecone index {self.index_name} already exists, checking dimension...")
                    self._check_and_fix_index_dimension()
                
                self.index = self.pc.Index(self.index_name)
                self.pinecone_available = True
                logger.info("Pinecone initialized successfully")
                break
                
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    logger.warning("All Pinecone initialization attempts failed. Falling back to in-memory storage")
        
        if not self.pinecone_available:
            logger.info("Using in-memory storage for memory management")
    
    def _create_index(self):
        """Create Pinecone index if it doesn't exist."""
        try:
            # Get embedding dimension based on model
            embedding_dimension = self._get_embedding_dimension()
            
            # Create index with serverless spec
            self.pc.create_index(
                name=self.index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            logger.info(f"Created Pinecone index: {self.index_name} with dimension {embedding_dimension}")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension based on the model being used."""
        embedding_model = settings.embedding_model
        # Map models to their dimensions
        model_dimensions = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        
        dimension = model_dimensions.get(embedding_model, 1536)  # Default to 1536
        logger.info(f"Using embedding dimension {dimension} for model {embedding_model}")
        return dimension
    
    def store_memory(
        self, 
        content: str, 
        session_id: str, 
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory in the vector database."""
        try:
            memory_id = str(uuid.uuid4())
            
            if self.pinecone_available:
                try:
                    # Generate embedding
                    embedding = self.openai_service.get_embedding(content)
                    
                    # Prepare metadata
                    memory_metadata = {
                        "content": content,
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        **(metadata or {})
                    }
                    
                    if user_id:
                        memory_metadata["user_id"] = user_id
                    
                    # Remove null values from metadata (Pinecone doesn't accept null)
                    memory_metadata = {k: v for k, v in memory_metadata.items() if v is not None}
                    
                    # Store in Pinecone
                    self.index.upsert(
                        vectors=[(memory_id, embedding, memory_metadata)]
                    )
                    
                    logger.info(f"Stored memory {memory_id} in Pinecone for session {session_id}")
                    return memory_id
                    
                except Exception as pinecone_error:
                    logger.error(f"Failed to store memory in Pinecone: {pinecone_error}")
                    logger.warning("Falling back to in-memory storage for this memory")
                    # Fall through to in-memory storage
                    self.pinecone_available = False  # Disable Pinecone for future operations
            
            # Store in memory (either as fallback or primary storage)
            self._in_memory_storage[memory_id] = {
                "content": content,
                "session_id": session_id,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            logger.info(f"Stored memory {memory_id} in memory for session {session_id}")
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    def retrieve_memories(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on semantic similarity."""
        try:
            limit = limit or self.max_retrieve
            threshold = threshold or self.threshold
            
            if self.pinecone_available:
                try:
                    # Generate query embedding
                    query_embedding = self.openai_service.get_embedding(query)
                    
                    # Prepare filter
                    filter_dict = {}
                    if session_id:
                        filter_dict["session_id"] = session_id
                    if user_id:
                        filter_dict["user_id"] = user_id
                    
                    # Query Pinecone with new API syntax
                    results = self.index.query(
                        vector=query_embedding,
                        top_k=limit,
                        include_metadata=True,
                        filter=filter_dict if filter_dict else None
                    )
                    
                    # Filter by threshold and format results
                    memories = []
                    for match in results.matches:
                        if match.score >= threshold:
                            memory = {
                                "id": match.id,
                                "content": match.metadata.get("content", ""),
                                "session_id": match.metadata.get("session_id", ""),
                                "user_id": match.metadata.get("user_id"),
                                "timestamp": match.metadata.get("timestamp"),
                                "similarity_score": match.score,
                                "metadata": {k: v for k, v in match.metadata.items() 
                                           if k not in ["content", "session_id", "user_id", "timestamp"]}
                            }
                            memories.append(memory)
                    
                    logger.info(f"Retrieved {len(memories)} memories from Pinecone for query: {query[:50]}...")
                    return memories
                    
                except Exception as pinecone_error:
                    logger.error(f"Failed to retrieve memories from Pinecone: {pinecone_error}")
                    logger.warning("Falling back to in-memory storage for memory retrieval")
                    self.pinecone_available = False  # Disable Pinecone for future operations
                    # Fall through to in-memory storage
            
            # Simple in-memory search (either as fallback or primary storage)
            memories = []
            for memory_id, memory_data in self._in_memory_storage.items():
                if session_id and memory_data["session_id"] != session_id:
                    continue
                if user_id and memory_data.get("user_id") != user_id:
                    continue
                
                # Simple keyword matching for in-memory storage
                if query.lower() in memory_data["content"].lower():
                    memories.append({
                        "id": memory_id,
                        "content": memory_data["content"],
                        "session_id": memory_data["session_id"],
                        "user_id": memory_data.get("user_id"),
                        "timestamp": memory_data["timestamp"],
                        "similarity_score": 0.8,  # Default score for in-memory
                        "metadata": memory_data.get("metadata", {})
                    })
            
            # Limit results
            memories = memories[:limit]
            logger.info(f"Retrieved {len(memories)} memories from in-memory storage for query: {query[:50]}...")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory from the vector database."""
        try:
            if self.pinecone_available:
                self.index.delete(ids=[memory_id])
                logger.info(f"Deleted memory {memory_id} from Pinecone")
            else:
                if memory_id in self._in_memory_storage:
                    del self._in_memory_storage[memory_id]
                    logger.info(f"Deleted memory {memory_id} from in-memory storage")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False
    
    def delete_session_memories(self, session_id: str) -> int:
        """Delete all memories for a specific session."""
        try:
            if self.pinecone_available:
                # Get the correct embedding dimension
                embedding_dimension = self._get_embedding_dimension()
                
                # Query to find all memories for the session
                results = self.index.query(
                    vector=[0] * embedding_dimension,  # Use correct dimension
                    top_k=10000,
                    include_metadata=True,
                    filter={"session_id": session_id}
                )
                
                # Delete all found memories
                if results.matches:
                    memory_ids = [match.id for match in results.matches]
                    self.index.delete(ids=memory_ids)
                    logger.info(f"Deleted {len(memory_ids)} memories from Pinecone for session {session_id}")
                    return len(memory_ids)
            else:
                # Delete from in-memory storage
                deleted_count = 0
                memory_ids_to_delete = []
                for memory_id, memory_data in self._in_memory_storage.items():
                    if memory_data["session_id"] == session_id:
                        memory_ids_to_delete.append(memory_id)
                        deleted_count += 1
                
                for memory_id in memory_ids_to_delete:
                    del self._in_memory_storage[memory_id]
                
                logger.info(f"Deleted {deleted_count} memories from in-memory storage for session {session_id}")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting session memories: {e}")
            return 0
    
    def get_memory_stats(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        try:
            if self.pinecone_available:
                filter_dict = {}
                if session_id:
                    filter_dict["session_id"] = session_id
                if user_id:
                    filter_dict["user_id"] = user_id
                
                # Get the correct embedding dimension
                embedding_dimension = self._get_embedding_dimension()
                
                # Query to get all memories (with dummy vector)
                results = self.index.query(
                    vector=[0] * embedding_dimension,  # Use correct dimension
                    top_k=10000,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )
                
                total_memories = len(results.matches)
                
                # Calculate average similarity if memories exist
                avg_similarity = 0
                if total_memories > 0:
                    avg_similarity = sum(match.score for match in results.matches) / total_memories
            else:
                # Count in-memory storage
                total_memories = 0
                for memory_data in self._in_memory_storage.values():
                    if session_id and memory_data["session_id"] != session_id:
                        continue
                    if user_id and memory_data.get("user_id") != user_id:
                        continue
                    total_memories += 1
                
                avg_similarity = 0.8  # Default for in-memory storage
            
            return {
                "total_memories": total_memories,
                "average_similarity": avg_similarity,
                "session_id": session_id,
                "user_id": user_id,
                "storage_type": "pinecone" if self.pinecone_available else "in-memory"
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"total_memories": 0, "average_similarity": 0, "storage_type": "error"}
    
    def _check_and_fix_index_dimension(self):
        """Check if the existing index has the correct dimension and recreate if needed."""
        try:
            # Get index description to check dimension
            index_description = self.pc.describe_index(self.index_name)
            current_dimension = index_description.dimension
            
            # Get expected dimension
            expected_dimension = self._get_embedding_dimension()
            
            if current_dimension != expected_dimension:
                logger.warning(f"Index dimension mismatch! Current: {current_dimension}, Expected: {expected_dimension}")
                logger.info(f"Recreating index {self.index_name} with correct dimension...")
                
                # Delete the old index
                self.pc.delete_index(self.index_name)
                logger.info(f"Deleted old index {self.index_name}")
                
                # Wait for deletion to complete
                time.sleep(5)
                
                # Create new index with correct dimension
                self._create_index()
                logger.info(f"Created new index {self.index_name} with dimension {expected_dimension}")
                
                # Wait for index to be ready
                time.sleep(2)
            else:
                logger.info(f"Index dimension is correct: {current_dimension}")
                
        except Exception as e:
            logger.error(f"Error checking/fixing index dimension: {e}")
            # If we can't check, assume it's okay and continue
            pass 