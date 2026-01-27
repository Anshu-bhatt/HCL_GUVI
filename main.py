# main.py

from fastapi import FastAPI
from config import config
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Multi-language voice authenticity detection",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "service": "AI Voice Detection API",
        "version": "1.0.0",
        "status": "running",
        "supported_languages": config.SUPPORTED_LANGUAGES
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": config.ENVIRONMENT
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Import string format for reload
        host=config.HOST, 
        port=config.PORT,
        reload=True  # Auto-reload on code changes
    )