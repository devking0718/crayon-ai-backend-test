from pydantic import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    # Pinecone Configuration
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "chatbot-memory")
    
    # Application Configuration
    app_name: str = "AI Chatbot with Memory"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Memory Configuration
    memory_similarity_threshold: float = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.7"))
    max_memories_to_retrieve: int = int(os.getenv("MAX_MEMORIES_TO_RETRIEVE", "5"))
    
    
    class Config:
        env_file = ".env"

settings = Settings() 