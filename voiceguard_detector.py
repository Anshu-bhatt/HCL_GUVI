# voiceguard_detector.py
"""
VoiceGUARD: AI Voice Detection using pre-trained Wav2Vec2 model
Fine-tuned for deepfake/AI-generated voice detection

Model: Mrkomiljon/voiceGUARD (HuggingFace Hub)
"""

import torch
import numpy as np
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from typing import Dict, Tuple, Optional, Any
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceGUARDDetector:
    """
    VoiceGUARD: AI Voice Detection using Wav2Vec2 embeddings + acoustic analysis
    
    Uses a hybrid approach:
    1. Wav2Vec2 embeddings for deep feature extraction
    2. Acoustic feature analysis for AI patterns detection
    
    AI-generated voices typically exhibit:
    - Lower temporal variance in embeddings
    - More consistent spectral patterns
    - Smoother pitch transitions
    - Less natural micro-variations
    """
    
    # Model configuration
    MODEL_NAME = "Mrkomiljon/voiceGUARD"
    SAMPLE_RATE = 16000
    MAX_DURATION = 10  # seconds
    
    LABEL_DESCRIPTIONS = {
        "HUMAN": "Bonafide human voice - authentic recording",
        "AI_GENERATED": "Synthetic/AI-generated voice - potential deepfake"
    }
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize VoiceGUARD detector
        
        Args:
            model_name: HuggingFace model identifier (default: Mrkomiljon/voiceGUARD)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name or self.MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = None
        self.model = None
        self._is_loaded = False
        
        logger.info(f"VoiceGUARD initialized (device: {self.device})")
        logger.info(f"Model: {self.model_name}")
    
    def load_model(self) -> None:
        """
        Load the VoiceGUARD model and processor
        Downloads from HuggingFace Hub if not cached
        """
        if self._is_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"Loading Wav2Vec2 model: {self.model_name}")
            logger.info("This may take a moment on first run (downloading model)...")
            
            # Load processor
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            
            # Load classification model
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
            
            # Move to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            self._is_loaded = True
            logger.info(f"✓ Wav2Vec2 model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def preprocess_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Preprocess audio for VoiceGUARD model
        
        Args:
            audio: Audio array (mono)
            sample_rate: Original sample rate
            
        Returns:
            Preprocessed audio array at 16kHz
        """
        # Resample if necessary
        if sample_rate != self.SAMPLE_RATE:
            logger.info(f"Resampling from {sample_rate}Hz to {self.SAMPLE_RATE}Hz")
            audio = librosa.resample(
                audio, 
                orig_sr=sample_rate, 
                target_sr=self.SAMPLE_RATE
            )
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / (max_val + 1e-8)
        
        # Trim silence
        try:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        except Exception:
            pass  # Keep original if trimming fails
        
        # Check duration limits
        max_samples = self.MAX_DURATION * self.SAMPLE_RATE
        if len(audio) > max_samples:
            logger.warning(f"Audio truncated to {self.MAX_DURATION}s")
            audio = audio[:max_samples]
        
        # Minimum duration check (0.5 seconds)
        min_samples = int(0.5 * self.SAMPLE_RATE)
        if len(audio) < min_samples:
            raise ValueError(f"Audio too short: {len(audio)/self.SAMPLE_RATE:.2f}s (min 0.5s)")
        
        return audio
    
    def classify(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Classify audio as AI-generated or Human
        
        Args:
            audio: Audio array (mono, float32)
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary containing:
                - classification: "AI_GENERATED" or "HUMAN"
                - confidence: Confidence score (0-1)
                - probabilities: Dict with probability for each class
                - description: Human-readable description
                - details: Additional analysis details
        """
        # Ensure model is loaded
        if not self._is_loaded:
            self.load_model()
        
        # Preprocess audio
        audio = self.preprocess_audio(audio, sample_rate)
        
        # Get audio duration for reporting
        duration = len(audio) / self.SAMPLE_RATE
        
        # Process through model
        inputs = self.processor(
            audio,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference - Model has 7 classes based on official demo:
        # 0: diffwave, 1: melgan, 2: parallel_wave_gan, 3: Real (Human)
        # 4: wavegrad, 5: wavnet, 6: wavernn
        # Class 3 = Real Human Voice, all others = AI-generated
        REAL_CLASS_ID = 3
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply softmax for probabilities over all 7 classes
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probabilities = probabilities.squeeze().cpu().numpy()
            
            # Get the predicted class
            predicted_class = int(torch.argmax(logits, dim=-1).item())
        
        # Binary classification: Class 3 = HUMAN, all others = AI_GENERATED
        prob_human = float(probabilities[REAL_CLASS_ID])
        prob_ai = float(1.0 - prob_human)  # Sum of all AI classes
        
        if predicted_class == REAL_CLASS_ID:
            classification = "HUMAN"
            confidence = prob_human
        else:
            classification = "AI_GENERATED"
            confidence = prob_ai
        
        # Create probability dict
        prob_dict = {
            "HUMAN": round(prob_human, 4),
            "AI_GENERATED": round(prob_ai, 4)
        }
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence)
        
        # Build result
        result = {
            "classification": classification,
            "confidence": round(confidence, 4),
            "confidence_level": confidence_level,
            "probabilities": prob_dict,
            "description": self.LABEL_DESCRIPTIONS[classification],
            "details": {
                "model": self.model_name,
                "audio_duration_seconds": round(duration, 2),
                "sample_rate": self.SAMPLE_RATE,
                "device": self.device,
                "threshold_used": 0.5,
                "predicted_class_id": predicted_class,
                "raw_scores": {
                    "bonafide_score": prob_dict["HUMAN"],
                    "spoof_score": prob_dict["AI_GENERATED"]
                }
            }
        }
        
        logger.info(
            f"Classification: {classification} "
            f"(confidence: {confidence:.2%}, level: {confidence_level})"
        )
        
        return result
    
    def _analyze_for_ai_patterns(self, audio: np.ndarray, embeddings: np.ndarray) -> float:
        """
        Analyze audio and embeddings for AI-generated patterns.
        
        AI-generated audio typically has:
        - Lower temporal variance in embeddings
        - More consistent/smooth patterns
        - Less micro-variations in pitch and energy
        - Different spectral characteristics
        
        Args:
            audio: Raw audio array
            embeddings: Wav2Vec2 embeddings (time x features)
            
        Returns:
            AI score between 0 (human) and 1 (AI-generated)
        """
        scores = []
        
        # 1. Embedding temporal variance analysis
        # AI audio tends to have more consistent/less variable embeddings
        temporal_variance = np.var(embeddings, axis=0).mean()
        # Normalize to 0-1 range (empirically tuned thresholds)
        variance_score = 1.0 - min(temporal_variance / 0.5, 1.0)
        scores.append(variance_score * 0.25)
        
        # 2. Embedding smoothness (consecutive frame similarity)
        if len(embeddings) > 1:
            frame_diffs = np.diff(embeddings, axis=0)
            smoothness = 1.0 / (1.0 + np.std(frame_diffs))
            # Higher smoothness = more likely AI
            smoothness_score = min(smoothness * 2, 1.0)
            scores.append(smoothness_score * 0.25)
        else:
            scores.append(0.25)
        
        # 3. Audio energy variance
        # AI audio often has more consistent energy levels
        frame_length = int(0.025 * self.SAMPLE_RATE)  # 25ms frames
        hop_length = int(0.010 * self.SAMPLE_RATE)    # 10ms hop
        
        # Calculate frame energies
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        if n_frames > 1:
            energies = []
            for i in range(n_frames):
                start = i * hop_length
                frame = audio[start:start + frame_length]
                energies.append(np.sum(frame ** 2))
            
            energy_variance = np.var(energies) / (np.mean(energies) + 1e-8)
            # Low variance = more likely AI
            energy_score = 1.0 - min(energy_variance / 2.0, 1.0)
            scores.append(energy_score * 0.25)
        else:
            scores.append(0.25)
        
        # 4. Zero-crossing rate variance
        # Human speech has more variable zero-crossing rates
        if len(audio) > frame_length:
            zcr_values = []
            for i in range(n_frames):
                start = i * hop_length
                frame = audio[start:start + frame_length]
                zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
                zcr_values.append(zcr)
            
            zcr_variance = np.var(zcr_values)
            # Low variance = more likely AI
            zcr_score = 1.0 - min(zcr_variance / 0.01, 1.0)
            scores.append(zcr_score * 0.25)
        else:
            scores.append(0.25)
        
        # Combine scores
        ai_score = sum(scores)
        
        # Apply sigmoid to get smoother distribution
        ai_score = 1.0 / (1.0 + np.exp(-5 * (ai_score - 0.5)))
        
        return float(ai_score)
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        Get human-readable confidence level
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Confidence level string
        """
        if confidence >= 0.95:
            return "VERY_HIGH"
        elif confidence >= 0.85:
            return "HIGH"
        elif confidence >= 0.70:
            return "MEDIUM"
        elif confidence >= 0.55:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def classify_file(self, file_path: str) -> Dict[str, Any]:
        """
        Classify an audio file
        
        Args:
            file_path: Path to audio file (WAV, MP3, FLAC, etc.)
            
        Returns:
            Classification result dictionary
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        logger.info(f"Loading audio file: {file_path}")
        
        # Load audio using librosa
        audio, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, mono=True)
        
        logger.info(f"Loaded: {len(audio)/sr:.2f}s, {sr}Hz")
        
        # Classify
        result = self.classify(audio, sr)
        result["details"]["file_path"] = file_path
        
        return result
    
    def classify_base64(self, audio_base64: str) -> Dict[str, Any]:
        """
        Classify Base64 encoded audio
        
        Args:
            audio_base64: Base64 encoded audio string
            
        Returns:
            Classification result dictionary
        """
        import base64
        import io
        
        # Decode Base64
        audio_bytes = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load audio
        audio, sr = librosa.load(audio_buffer, sr=self.SAMPLE_RATE, mono=True)
        
        # Classify
        result = self.classify(audio, sr)
        result["details"]["input_type"] = "base64"
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "device": self.device,
            "sample_rate": self.SAMPLE_RATE,
            "max_duration_seconds": self.MAX_DURATION,
            "supported_labels": ["HUMAN", "AI_GENERATED"],
            "label_descriptions": self.LABEL_DESCRIPTIONS
        }


# Singleton instance for API usage
_detector_instance: Optional[VoiceGUARDDetector] = None


def get_detector() -> VoiceGUARDDetector:
    """
    Get or create VoiceGUARD detector singleton
    
    Returns:
        VoiceGUARDDetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = VoiceGUARDDetector()
        _detector_instance.load_model()
    
    return _detector_instance


# CLI testing
if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("VOICEGUARD DETECTOR TEST")
    print("=" * 60 + "\n")
    
    # Initialize detector
    detector = VoiceGUARDDetector()
    
    # Print model info
    print("Model Information:")
    info = detector.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nLoading model...")
    detector.load_model()
    
    # Test with synthetic audio if no file provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\nClassifying: {file_path}")
        result = detector.classify_file(file_path)
    else:
        print("\nTesting with synthetic audio (sine wave)...")
        # Generate 2 seconds of 440Hz sine wave
        duration = 2.0
        t = np.linspace(0, duration, int(16000 * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        result = detector.classify(audio, 16000)
    
    print("\n" + "-" * 40)
    print("CLASSIFICATION RESULT")
    print("-" * 40)
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Confidence Level: {result['confidence_level']}")
    print(f"\nProbabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.2%}")
    print(f"\nDescription: {result['description']}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60 + "\n")
