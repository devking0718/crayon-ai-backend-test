from openai import OpenAI
from typing import List, Dict, Any, Optional
from app.core.config import settings
import tiktoken
import logging

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature
        # Use the newer embedding model
        self.embedding_model = settings.embedding_model
        
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI's embedding model."""
        try:
            logger.info(f"Generating embedding with model: {self.embedding_model}")
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            logger.info(f"Successfully generated embedding with model: {self.embedding_model}")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding with model {self.embedding_model}: {e}")
            raise
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        use_memory: bool = True,
        memories: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate chat completion with optional memory context."""
        try:
            # Prepare system message
            system_message = self._build_system_message(use_memory, memories)
            
            # Add system message to the beginning
            full_messages = [{"role": "system", "content": system_message}] + messages
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return {
                "response": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    def should_use_memory(self, message: str, conversation_history: List[Dict[str, str]]) -> bool:
        """AI decides whether to use memory based on the current message and context."""
        try:
            decision_prompt = f"""
            You are an AI assistant that decides whether to retrieve memories for a conversation.
            
            Current message: "{message}"
            
            Recent conversation context:
            {self._format_conversation_history(conversation_history[-5:])}
            
            Decision criteria:
            - Use memory if the message asks about past information, references previous topics, or needs context
            - Use memory if the user is asking follow-up questions or continuing a previous discussion
            - Don't use memory for simple greetings, general questions, or standalone queries
            
            Respond with only "YES" or "NO" and a brief reason.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,  # Use the configured model from settings
                messages=[{"role": "user", "content": decision_prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            decision = response.choices[0].message.content.strip().upper()
            return decision.startswith("YES")
            
        except Exception as e:
            logger.error(f"Error in memory decision: {e}")
            return True  # Default to using memory if decision fails
    
    def _build_system_message(self, use_memory: bool, memories: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build the system message with optional memory context."""
        base_message = """You are a helpful AI assistant with memory capabilities. You can remember past conversations and use that context to provide more personalized and relevant responses."""
        
        if use_memory and memories:
            memory_context = "\n\nRelevant memories from past conversations:\n"
            for i, memory in enumerate(memories, 1):
                memory_context += f"{i}. {memory.get('content', '')}\n"
            
            base_message += memory_context
            base_message += "\nUse these memories to provide contextually relevant responses when appropriate."
        
        return base_message
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for decision making."""
        formatted = ""
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100]  # Truncate for decision making
            formatted += f"{role}: {content}\n"
        return formatted
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except:
            # Fallback to approximate counting
            return len(text.split()) * 1.3 