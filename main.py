from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chat, memory
from app.core.config import settings

app = FastAPI(
    title="AI Chatbot with Memory",
    description="A production-ready chatbot that remembers conversations using vector storage",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(memory.router, prefix="/api/v1", tags=["memory"])

@app.get("/")
async def root():
    return {"message": "AI Chatbot with Memory API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "chatbot-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 