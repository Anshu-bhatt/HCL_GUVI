# main.py
"""
AI Voice Detection API
VoiceGUARD: Detect AI-generated vs Human voice using Wav2Vec2
"""

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from config import config
from api_models import (
    ClassifyRequest, 
    ClassifyResponse, 
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse
)
from voiceguard_detector import VoiceGUARDDetector, get_detector
import uvicorn
import logging
import io
import librosa
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global detector instance
detector: VoiceGUARDDetector = None


# Simple response model for file upload
class SimpleClassifyResponse(BaseModel):
    """Simplified classification response"""
    classification: str
    confidence: float
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "classification": "AI_GENERATED",
                "confidence": 0.95
            }
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler
    Loads model on startup, cleans up on shutdown
    """
    global detector
    
    # Startup: Load the VoiceGUARD model
    logger.info("=" * 50)
    logger.info("Starting AI Voice Detection API...")
    logger.info("=" * 50)
    
    try:
        detector = VoiceGUARDDetector()
        detector.load_model()
        logger.info("✓ VoiceGUARD model loaded and ready")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will attempt to load model on first request")
        detector = VoiceGUARDDetector()
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down AI Voice Detection API...")
    detector = None


# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="AI Voice Detection API",
    description="""
    ## VoiceGUARD - AI Voice Detection System
    
    Multi-language voice authenticity detection API that identifies whether 
    audio is **AI-generated** (deepfake) or **Human** (authentic).
    
    ### Features
    - 🎯 Pre-trained Wav2Vec2 model fine-tuned for deepfake detection
    - 🌍 Multi-language support (Tamil, English, Hindi, Malayalam, Telugu)
    - 📊 Confidence scores and detailed analysis
    - 🔒 Base64 audio input support
    
    ### Supported Audio Formats
    - MP3, WAV, FLAC, OGG, M4A
    
    ### Endpoints
    - **POST /detect** - Upload audio file directly (recommended)
    - **POST /classify** - Send Base64 encoded audio
    
    ### Model
    Uses `Mrkomiljon/voiceGUARD` from HuggingFace Hub
    """,
    version="2.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint - API information
    
    Returns basic API information and available endpoints.
    """
    return {
        "service": "AI Voice Detection API",
        "version": "2.0.0",
        "model": "VoiceGUARD (Mrkomiljon/voiceGUARD)",
        "status": "running",
        "supported_languages": config.SUPPORTED_LANGUAGES,
        "supported_formats": ["MP3", "WAV", "FLAC", "OGG", "M4A"],
        "endpoints": {
            "detect": "/detect - POST - Upload audio file directly (recommended)",
            "classify": "/classify - POST - Classify Base64 audio",
            "health": "/health - GET - Health check",
            "model_info": "/model/info - GET - Model information",
            "docs": "/docs - API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the API and model.
    """
    global detector
    
    model_loaded = detector is not None and detector._is_loaded
    
    return HealthResponse(
        status="healthy",
        environment=config.ENVIRONMENT,
        model_loaded=model_loaded,
        model_name=detector.model_name if detector else "Not initialized",
        device=detector.device if detector else "N/A"
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get model information
    
    Returns detailed information about the loaded VoiceGUARD model.
    """
    global detector
    
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    info = detector.get_model_info()
    return ModelInfoResponse(**info)


@app.post(
    "/detect",
    response_model=SimpleClassifyResponse,
    responses={
        200: {"description": "Successful classification"},
        400: {"model": ErrorResponse, "description": "Invalid audio file"},
        500: {"model": ErrorResponse, "description": "Classification failed"}
    },
    tags=["Classification"],
    summary="Upload audio file for detection"
)
async def detect_audio_file(
    file: UploadFile = File(..., description="Audio file (MP3, WAV, FLAC, OGG, M4A)")
):
    """
    🎯 **Upload audio file directly for AI voice detection**
    
    This is the recommended endpoint for easy audio classification.
    Simply upload your audio file and get the result.
    
    ## Supported Formats
    - MP3, WAV, FLAC, OGG, M4A
    
    ## Response
    - **classification**: `AI_GENERATED` or `HUMAN`
    - **confidence**: Confidence score (0.0 - 1.0)
    
    ## Example using curl
    ```bash
    curl -X POST "http://localhost:8000/detect" \\
      -F "file=@your_audio.mp3"
    ```
    """
    global detector
    
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized. Please try again."
        )
    
    # Validate file type
    allowed_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma']
    file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(allowed_extensions)}"
        )
    
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Load audio from bytes
        audio_buffer = io.BytesIO(content)
        audio, sr = librosa.load(audio_buffer, sr=16000, mono=True)
        
        logger.info(f"Audio loaded: {len(audio)/sr:.2f}s, {sr}Hz")
        
        # Classify the audio
        result = detector.classify(audio, sr)
        
        # Log result
        logger.info(
            f"Classification complete: {result['classification']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        
        # Return simplified response
        return SimpleClassifyResponse(
            classification=result['classification'],
            confidence=round(result['confidence'], 4)
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={
        200: {"description": "Successful classification"},
        400: {"model": ErrorResponse, "description": "Invalid audio input"},
        500: {"model": ErrorResponse, "description": "Classification failed"}
    },
    tags=["Classification"]
)
async def classify_audio(request: ClassifyRequest):
    """
    Classify audio as AI-generated or Human
    
    ## Input
    - **audio_base64**: Base64 encoded audio file (MP3, WAV, FLAC, OGG, M4A)
    - **language**: Optional language hint (not used for classification, for logging only)
    
    ## Output
    - **classification**: `AI_GENERATED` or `HUMAN`
    - **confidence**: Confidence score (0.0 - 1.0)
    - **confidence_level**: `VERY_HIGH`, `HIGH`, `MEDIUM`, `LOW`, or `VERY_LOW`
    - **probabilities**: Probability for each class
    - **description**: Human-readable description
    - **details**: Additional analysis details
    
    ## Example
    ```python
    import requests
    import base64
    
    # Read and encode audio file
    with open("audio.mp3", "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    
    # Make request
    response = requests.post(
        "http://localhost:8000/classify",
        json={"audio_base64": audio_base64}
    )
    print(response.json())
    ```
    """
    global detector
    
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized. Please try again."
        )
    
    try:
        # Log request
        if request.language:
            logger.info(f"Classification request - Language hint: {request.language}")
        else:
            logger.info("Classification request received")
        
        # Classify the audio
        result = detector.classify_base64(request.audio_base64)
        
        # Log result
        logger.info(
            f"Classification complete: {result['classification']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        
        return ClassifyResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


# Legacy endpoint for backward compatibility
@app.get("/api/health", tags=["Legacy"])
async def legacy_health():
    """Legacy health endpoint (deprecated)"""
    return await health_check()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST, 
        port=config.PORT,
        reload=True
    )