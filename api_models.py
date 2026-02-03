# api_models.py
"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List, Any
from enum import Enum


class ClassificationLabel(str, Enum):
    """Voice classification labels"""
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"


class ConfidenceLevel(str, Enum):
    """Confidence level categories"""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class ClassifyRequest(BaseModel):
    """Request model for /classify endpoint"""
    
    audio_base64: str = Field(
        ...,
        description="Base64 encoded audio file (MP3, WAV, FLAC, OGG supported)",
        min_length=100
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language hint (Tamil, English, Hindi, Malayalam, Telugu)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "audio_base64": "UklGRiQA...(base64 encoded audio)...",
                "language": "English"
            }
        }
    )


class ClassificationDetails(BaseModel):
    """Detailed classification information"""
    
    model: str = Field(..., description="Model used for classification")
    audio_duration_seconds: float = Field(..., description="Duration of processed audio")
    sample_rate: int = Field(..., description="Sample rate used for processing")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    threshold_used: float = Field(default=0.5, description="Classification threshold")
    raw_scores: Dict[str, float] = Field(..., description="Raw model output scores")


class ClassifyResponse(BaseModel):
    """Response model for /classify endpoint"""
    
    classification: ClassificationLabel = Field(
        ...,
        description="Classification result: AI_GENERATED or HUMAN"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Human-readable confidence level"
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability for each class"
    )
    description: str = Field(
        ...,
        description="Human-readable description of the classification"
    )
    details: ClassificationDetails = Field(
        ...,
        description="Additional classification details"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "classification": "AI_GENERATED",
                "confidence": 0.9234,
                "confidence_level": "VERY_HIGH",
                "probabilities": {
                    "HUMAN": 0.0766,
                    "AI_GENERATED": 0.9234
                },
                "description": "Synthetic/AI-generated voice - potential deepfake",
                "details": {
                    "model": "Mrkomiljon/voiceGUARD",
                    "audio_duration_seconds": 3.5,
                    "sample_rate": 16000,
                    "device": "cpu",
                    "threshold_used": 0.5,
                    "raw_scores": {
                        "bonafide_score": 0.0766,
                        "spoof_score": 0.9234
                    }
                }
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Invalid audio format",
                "details": "Audio must be at least 0.5 seconds long"
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    environment: str = Field(..., description="Running environment")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: str = Field(..., description="Name of the loaded model")
    device: str = Field(..., description="Device being used")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    
    model_name: str
    is_loaded: bool
    device: str
    sample_rate: int
    max_duration_seconds: int
    supported_labels: List[str]
    label_descriptions: Dict[str, str]
