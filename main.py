# main.py
# Updated: Fixed duration-aware detection for short audio files

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config import config
import uvicorn
import time
import logging
from contextlib import asynccontextmanager

from schemas import (
    VoiceDetectionRequest, 
    VoiceDetectionResponse, 
    ErrorResponse,
    HealthCheckResponse
)
from audio_preprocessor import AudioProcessor
from model_detector import HybridDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances (loaded on startup)
audio_processor = None
hybrid_detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - loads models on startup"""
    global audio_processor, hybrid_detector
    
    logger.info("🚀 Starting AI Voice Detection API...")
    logger.info("Loading models...")
    
    try:
        # Initialize audio processor
        audio_processor = AudioProcessor(sample_rate=16000, max_duration=30)
        logger.info("✓ Audio processor loaded")
        
        # Initialize hybrid detector (loads Wav2Vec2)
        hybrid_detector = HybridDetector(ai_threshold=0.5)  # Default threshold
        logger.info("✓ Hybrid detector loaded")
        
        logger.info("✅ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AI Voice Detection API",
    description="Multi-language voice authenticity detection using Wav2Vec2 + Acoustic Features",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler for validation errors
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            status="error",
            message=str(exc),
            details={"error_type": "ValidationError"}
        ).model_dump()
    )


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "service": "AI Voice Detection API",
        "version": "1.0.0",
        "status": "running",
        "supported_languages": config.SUPPORTED_LANGUAGES,
        "endpoints": {
            "detection": "/api/voice-detection",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        environment=config.ENVIRONMENT,
        models_loaded=audio_processor is not None and hybrid_detector is not None,
        supported_languages=config.SUPPORTED_LANGUAGES
    )


@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def detect_voice(request: VoiceDetectionRequest):
    """
    Detect if audio is AI-generated or human
    
    This endpoint analyzes audio using:
    1. Wav2Vec2 embeddings (pre-trained transformer)
    2. Acoustic features (MFCC, spectral, temporal)
    3. Hybrid detection (combines both approaches)
    
    Supports: Tamil, English, Hindi, Malayalam, Telugu
    """
    start_time = time.time()
    
    try:
        # Validate models are loaded
        if audio_processor is None or hybrid_detector is None:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Please try again in a moment."
            )
        
        logger.info(f"Processing {request.language} audio ({len(request.audioBase64)} chars)")
        
        # Step 1: Decode and load audio (validation is done inside decode_base64_audio)
        try:
            audio, sample_rate = audio_processor.decode_base64_audio(request.audioBase64)
            logger.info(f"✓ Audio decoded: {len(audio)} samples at {sample_rate}Hz")
        except Exception as e:
            raise ValueError(f"Failed to decode audio: {str(e)}")
        
        # Calculate original duration before preprocessing (preprocessing may pad audio)
        original_duration = len(audio) / sample_rate
        
        # Step 2: Preprocess audio
        try:
            audio_processed = audio_processor.preprocess_audio(audio)
            logger.info(f"✓ Audio preprocessed: {len(audio_processed)} samples")
        except Exception as e:
            raise ValueError(f"Failed to preprocess audio: {str(e)}")
        
        # Step 3: Extract acoustic features
        try:
            acoustic_features = audio_processor.extract_features(audio_processed)
            logger.info(f"✓ Extracted {len(acoustic_features)} acoustic features")
        except Exception as e:
            raise ValueError(f"Failed to extract features: {str(e)}")
        
        # Step 4: Run hybrid detection (pass original duration to handle short audio correctly)
        try:
            classification, confidence, details = hybrid_detector.detect(
                audio_processed,
                acoustic_features,
                sample_rate,
                original_duration=original_duration
            )
            confidence_level = details.get('confidence_level', 'UNKNOWN')
            logger.info(f"✓ Detection: {classification} ({confidence_level} confidence: {confidence:.2%})")
        except Exception as e:
            raise ValueError(f"Failed to run detection: {str(e)}")
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Generate explanation
        explanation = _generate_explanation(classification, confidence, confidence_level, details)
        
        # Build response
        response = VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence, 4),
            confidenceLevel=confidence_level,
            explanation=explanation,
            details={
                "wav2vec2_score": round(details.get('wav2vec2', {}).get('ai_score', 0), 4),
                "acoustic_score": round(details.get('acoustic_ai_score', 0), 4),
                "combined_score": round(details.get('combined_score', 0), 4),
                "processing_time_ms": processing_time_ms,
                "audio_duration_seconds": round(len(audio) / sample_rate, 2),
                "sample_rate": sample_rate
            }
        )
        
        logger.info(f"✅ Request completed in {processing_time_ms}ms")
        return response
        
    except ValueError as e:
        # Validation errors (400)
        logger.warning(f"Validation error: {e}")
        return VoiceDetectionResponse(
            status="error",
            message=str(e)
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        # Unexpected errors (500)
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


def _generate_explanation(classification: str, confidence: float, 
                         confidence_level: str, details: dict) -> str:
    """Generate human-readable explanation of the classification"""
    
    combined_score = details.get('combined_score', 0)
    wav2vec2_score = details.get('wav2vec2', {}).get('ai_score', 0)
    acoustic_score = details.get('acoustic_ai_score', 0)
    
    if classification == "UNCERTAIN":
        return (
            f"Classification uncertain (score: {combined_score:.2f}). "
            f"The audio falls in a borderline zone between AI and human characteristics. "
            f"Manual review or additional samples recommended."
        )
    
    if classification == "AI_GENERATED":
        reasons = []
        
        if wav2vec2_score > 0.7:
            reasons.append("high consistency in voice embeddings")
        
        if acoustic_score > 0.5:
            reasons.append("smooth acoustic patterns")
        
        if confidence_level == "HIGH":
            confidence_desc = "very high"
        elif confidence_level == "MEDIUM":
            confidence_desc = "moderate"
        else:
            confidence_desc = "low"
        
        if reasons:
            return f"Detected as AI-generated with {confidence_desc} confidence ({confidence:.0%}) due to {' and '.join(reasons)}."
        else:
            return f"Detected as AI-generated with {confidence_desc} confidence ({confidence:.0%})."
    
    else:  # HUMAN
        reasons = []
        
        if wav2vec2_score < 0.3:
            reasons.append("natural variations in voice embeddings")
        
        if acoustic_score < 0.3:
            reasons.append("irregular acoustic patterns")
        
        if confidence_level == "HIGH":
            confidence_desc = "very high"
        elif confidence_level == "MEDIUM":
            confidence_desc = "moderate"
        else:
            confidence_desc = "low"
        
        if reasons:
            return f"Detected as human voice with {confidence_desc} confidence ({confidence:.0%}) due to {' and '.join(reasons)}."
        else:
            return f"Detected as human voice with {confidence_desc} confidence ({confidence:.0%})."

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Import string format for reload
        host=config.HOST, 
        port=config.PORT,
        reload=True  # Auto-reload on code changes
    )