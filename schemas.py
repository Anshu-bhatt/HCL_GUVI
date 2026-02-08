# schemas.py

from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, Literal
import base64

class VoiceDetectionRequest(BaseModel):
    """Request schema for voice detection API"""
    
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ...,
        description="Language of the audio"
    )
    
    audioFormat: Literal["mp3", "wav"] = Field(
        default="mp3",
        description="Audio format (mp3 or wav)"
    )
    
    audioBase64: str = Field(
        ...,
        description="Base64 encoded audio data",
        min_length=100
    )
    
    @validator('audioBase64')
    def validate_base64(cls, v):
        """Validate Base64 encoding"""
        try:
            # Try to decode to verify it's valid Base64
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid Base64 encoding")
    
    class Config:
        json_schema_extra = {
            "example": {
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
            }
        }


class VoiceDetectionResponse(BaseModel):
    """Response schema for voice detection API"""
    
    status: Literal["success", "error"] = Field(
        ...,
        description="Status of the request"
    )
    
    language: Optional[str] = Field(
        None,
        description="Language of the audio"
    )
    
    classification: Optional[Literal["AI_GENERATED", "HUMAN", "UNCERTAIN"]] = Field(
        None,
        description="Classification result"
    )
    
    confidenceScore: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    
    confidenceLevel: Optional[Literal["HIGH", "MEDIUM", "LOW"]] = Field(
        None,
        description="Confidence level"
    )
    
    explanation: Optional[str] = Field(
        None,
        description="Explanation of the classification"
    )
    
    details: Optional[Dict] = Field(
        None,
        description="Additional detection details"
    )
    
    message: Optional[str] = Field(
        None,
        description="Error message (if status is error)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "language": "Tamil",
                "classification": "AI_GENERATED",
                "confidenceScore": 0.87,
                "confidenceLevel": "HIGH",
                "explanation": "High consistency in embeddings and smooth acoustic features suggest AI-generated voice",
                "details": {
                    "wav2vec2_score": 0.92,
                    "acoustic_score": 0.75,
                    "processing_time_ms": 1250
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    
    status: Literal["error"] = "error"
    message: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Audio too short: 0.5s (min 1.0s)",
                "details": {
                    "error_type": "ValidationError",
                    "audio_duration": 0.5
                }
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    
    status: str = Field(..., description="Health status")
    environment: str = Field(..., description="Environment name")
    models_loaded: bool = Field(..., description="Whether AI models are loaded")
    supported_languages: list = Field(..., description="List of supported languages")
