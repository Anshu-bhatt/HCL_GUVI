# model_detector.py

import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class Wav2Vec2Detector:
    """
    AI Voice Detection using pre-trained Wav2Vec2 embeddings
    NO FINE-TUNING NEEDED - Uses heuristic analysis
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        """
        Initialize Wav2Vec2 detector with pre-trained model
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading Wav2Vec2 model: {model_name}")
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        logger.info("✓ Wav2Vec2 model loaded successfully")
    
    def extract_embeddings(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract Wav2Vec2 embeddings from audio
        
        Args:
            audio: Audio array
            sample_rate: Sample rate (must be 16000 for Wav2Vec2)
            
        Returns:
            Embeddings array of shape (sequence_length, 768)
        """
        # Resample if needed
        if sample_rate != 16000:
            raise ValueError("Wav2Vec2 requires 16kHz sample rate")
        
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Extract features (no gradient needed for inference)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
        logger.info(f"Extracted embeddings: shape={embeddings.shape}")
        return embeddings
    
    def analyze_embeddings(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Analyze embeddings for AI-generated voice characteristics
        
        AI-generated voices typically show:
        - Lower variance across time
        - More consistent patterns
        - Smoother transitions
        
        Args:
            embeddings: Wav2Vec2 embeddings (sequence_length, 768)
            
        Returns:
            Dictionary of analysis metrics
        """
        # Calculate statistics
        temporal_variance = np.var(embeddings, axis=0).mean()
        temporal_std = np.std(embeddings, axis=0).mean()
        
        # Frame-to-frame differences (smoothness)
        frame_diffs = np.diff(embeddings, axis=0)
        smoothness = np.mean(np.abs(frame_diffs))
        
        # Consistency score (inverse of variance)
        consistency = 1.0 / (1.0 + temporal_variance)
        
        # Distribution analysis
        embedding_mean = np.mean(embeddings, axis=0)
        embedding_concentration = np.mean(np.abs(embedding_mean))
        
        metrics = {
            'temporal_variance': float(temporal_variance),
            'temporal_std': float(temporal_std),
            'smoothness': float(smoothness),
            'consistency': float(consistency),
            'concentration': float(embedding_concentration)
        }
        
        logger.info(f"Embedding analysis: {metrics}")
        return metrics
    
    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float, Dict]:
        """
        Detect if audio is AI-generated or human
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Tuple of (classification, confidence_score, details)
        """
        # Extract embeddings
        embeddings = self.extract_embeddings(audio, sample_rate)
        
        # Analyze embeddings
        metrics = self.analyze_embeddings(embeddings)
        
        # HEURISTIC RULES (no training needed!)
        # These thresholds can be tuned with test data
        ai_score = 0.0
        
        # Rule 1: High consistency suggests AI
        if metrics['consistency'] > 0.85:
            ai_score += 0.3
        
        # Rule 2: High smoothness suggests AI
        if metrics['smoothness'] < 0.05:
            ai_score += 0.3
        
        # Rule 3: Low variance suggests AI
        if metrics['temporal_variance'] < 0.01:
            ai_score += 0.2
        
        # Rule 4: High concentration suggests AI
        if metrics['concentration'] > 0.1:
            ai_score += 0.2
        
        # Classification
        classification = "AI_GENERATED" if ai_score > 0.5 else "HUMAN"
        confidence = ai_score if ai_score > 0.5 else (1.0 - ai_score)
        
        details = {
            'metrics': metrics,
            'ai_score': ai_score,
            'classification': classification,
            'confidence': confidence
        }
        
        logger.info(f"Detection result: {classification} (confidence: {confidence:.2f})")
        return classification, confidence, details


class HybridDetector:
    """
    Combines Wav2Vec2 embeddings + Acoustic features for better accuracy
    """
    
    def __init__(self):
        """Initialize hybrid detector"""
        self.wav2vec2_detector = Wav2Vec2Detector()
        logger.info("✓ HybridDetector initialized")
    
    def detect(self, audio: np.ndarray, acoustic_features: Dict[str, float], 
               sample_rate: int = 16000) -> Tuple[str, float, Dict]:
        """
        Hybrid detection using both Wav2Vec2 and acoustic features
        
        Args:
            audio: Audio array
            acoustic_features: Features from AudioProcessor
            sample_rate: Sample rate
            
        Returns:
            Tuple of (classification, confidence_score, details)
        """
        # Get Wav2Vec2 prediction
        w2v_class, w2v_conf, w2v_details = self.wav2vec2_detector.detect(audio, sample_rate)
        
        # Analyze acoustic features for AI patterns
        acoustic_ai_score = 0.0
        
        # Low variance in pitch suggests AI
        if acoustic_features.get('spectral_centroid_std', 100) < 50:
            acoustic_ai_score += 0.3
        
        # Consistent zero-crossing suggests AI
        if acoustic_features.get('zcr_std', 1.0) < 0.01:
            acoustic_ai_score += 0.3
        
        # Smooth RMS energy suggests AI
        if acoustic_features.get('rms_std', 1.0) < 0.05:
            acoustic_ai_score += 0.2
        
        # Low MFCC variance suggests AI
        mfcc_vars = [acoustic_features.get(f'mfcc_{i}_std', 10) for i in range(13)]
        if np.mean(mfcc_vars) < 5.0:
            acoustic_ai_score += 0.2
        
        # Combine scores (weighted average)
        w2v_score = w2v_conf if w2v_class == "AI_GENERATED" else (1.0 - w2v_conf)
        combined_score = 0.7 * w2v_score + 0.3 * acoustic_ai_score
        
        # Final classification
        classification = "AI_GENERATED" if combined_score > 0.5 else "HUMAN"
        confidence = combined_score if combined_score > 0.5 else (1.0 - combined_score)
        
        details = {
            'wav2vec2': w2v_details,
            'acoustic_ai_score': acoustic_ai_score,
            'combined_score': combined_score,
            'classification': classification,
            'confidence': confidence
        }
        
        logger.info(f"Hybrid detection: {classification} (confidence: {confidence:.2f})")
        return classification, confidence, details


if __name__ == "__main__":
    # Test the detector
    logging.basicConfig(level=logging.INFO)
    print("\n" + "="*60)
    print("TESTING WAV2VEC2 DETECTOR")
    print("="*60 + "\n")
    
    # Create synthetic test audio
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    print("Creating detector (will download model on first run)...")
    detector = Wav2Vec2Detector()
    
    print("\nTesting detection...")
    classification, confidence, details = detector.detect(audio, sample_rate)
    
    print(f"\n✓ Classification: {classification}")
    print(f"✓ Confidence: {confidence:.2%}")
    print(f"\n✓ Embedding metrics:")
    for key, value in details['metrics'].items():
        print(f"  - {key}: {value:.6f}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")
