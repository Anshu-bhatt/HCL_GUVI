# voiceguard_detector.py
"""
VoiceGUARD: AI Voice Detection using Audio Spectrogram Transformer (AST)
Fine-tuned for deepfake/AI-generated voice detection

Model: 012shin/KAIROS-ast-fake-audio-detection (HuggingFace Hub)
- Binary classification: Real (0) vs Fake (1)
- Based on Audio Spectrogram Transformer architecture
"""

import torch
import numpy as np
import librosa
from transformers import ASTForAudioClassification, ASTFeatureExtractor
from typing import Dict, Tuple, Optional, Any
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceGUARDDetector:
    """
    VoiceGUARD: AI Voice Detection using Audio Spectrogram Transformer (AST)
    
    Uses the KAIROS model which is specifically trained for:
    - Binary classification: Real vs Fake audio
    - Based on AST architecture for robust audio analysis
    
    The model analyzes spectral patterns to detect:
    - AI-generated voices (TTS, voice cloning, deepfakes)
    - Authentic human speech recordings
    """
    
    # Model configuration - Using KAIROS AST model for better accuracy
    MODEL_NAME = "012shin/KAIROS-ast-fake-audio-detection"
    EXTRACTOR_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    SAMPLE_RATE = 16000
    MAX_DURATION = 10  # seconds
    
    # Label mapping: Class 0 = Real, Class 1 = Fake
    REAL_CLASS_ID = 0
    FAKE_CLASS_ID = 1
    
    LABEL_DESCRIPTIONS = {
        "HUMAN": "Bonafide human voice - authentic recording",
        "AI_GENERATED": "Synthetic/AI-generated voice - potential deepfake"
    }
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize VoiceGUARD detector
        
        Args:
            model_name: HuggingFace model identifier (default: KAIROS AST model)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name or self.MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.extractor = None
        self.model = None
        self._is_loaded = False
        
        # For backward compatibility
        self.processor = None
        
        logger.info(f"VoiceGUARD initialized (device: {self.device})")
        logger.info(f"Model: {self.model_name}")
    
    def load_model(self) -> None:
        """
        Load the VoiceGUARD model and feature extractor
        Downloads from HuggingFace Hub if not cached
        """
        if self._is_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"Loading AST model: {self.model_name}")
            logger.info("This may take a moment on first run (downloading model)...")
            
            # Load feature extractor from AST base model
            self.extractor = ASTFeatureExtractor.from_pretrained(self.EXTRACTOR_NAME)
            
            # Load the KAIROS classification model
            self.model = ASTForAudioClassification.from_pretrained(self.model_name)
            
            # Move to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            self._is_loaded = True
            logger.info(f"✓ AST model loaded successfully on {self.device}")
            
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
        Classify audio as AI-generated or Human using lightweight ensemble approach
        
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
        
        # Process through AST feature extractor
        inputs = self.extractor(
            audio,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run AST model inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply softmax for probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probabilities = probabilities.squeeze().cpu().numpy()
        
        # AST model predictions
        ast_prob_real = float(probabilities[self.REAL_CLASS_ID])
        ast_prob_fake = float(probabilities[self.FAKE_CLASS_ID])
        
        # Lightweight acoustic feature analysis for ensemble
        acoustic_real_score = self._lightweight_acoustic_analysis(audio)
        
        # BIAS CORRECTION: The AST model tends to be biased toward AI detection
        # Apply calibration to balance predictions
        
        # Step 1: Recalibrate AST probabilities with temperature scaling
        # The model is overly confident in AI detection, so we soften the predictions
        temperature = 1.5  # Higher temperature = softer probabilities
        calibrated_ast_real = 1.0 / (1.0 + np.exp(-np.log(ast_prob_real / (ast_prob_fake + 1e-10)) / temperature))
        
        # Step 2: Apply human-bias correction
        # If acoustic features suggest human characteristics, boost human probability
        if acoustic_real_score > 0.4:  # Some human characteristics detected
            # Apply progressive boost based on acoustic score strength
            human_boost = min(0.3, (acoustic_real_score - 0.4) * 0.5)
            calibrated_ast_real = min(0.95, calibrated_ast_real + human_boost)
        
        # Step 3: Ensemble with adjusted weights favoring acoustic features for balance
        # Increase acoustic weight to 50% to counter AST bias
        ensemble_prob_real = 0.5 * calibrated_ast_real + 0.5 * acoustic_real_score
        ensemble_prob_fake = 1.0 - ensemble_prob_real
        
        # Step 4: Final threshold adjustment for balanced classification
        # Apply soft threshold that favors human detection when scores are close
        if 0.35 <= ensemble_prob_real <= 0.65:  # Uncertain region
            # In uncertain cases, slightly favor human classification to reduce bias
            ensemble_prob_real = ensemble_prob_real * 1.15  # 15% boost
            ensemble_prob_real = min(0.85, ensemble_prob_real)  # Cap at 85%
            ensemble_prob_fake = 1.0 - ensemble_prob_real
        
        # Step 5: Confidence calibration based on agreement
        # If both methods agree strongly, boost confidence
        agreement = abs(calibrated_ast_real - acoustic_real_score)
        if agreement < 0.25:  # Methods agree reasonably well
            if ensemble_prob_real > 0.5:
                ensemble_prob_real = min(0.92, ensemble_prob_real * 1.1)
            else:
                ensemble_prob_fake = min(0.92, ensemble_prob_fake * 1.1)
            ensemble_prob_fake = 1.0 - ensemble_prob_real
        
        # Final classification
        if ensemble_prob_real > 0.5:
            classification = "HUMAN"
            confidence = ensemble_prob_real
        else:
            classification = "AI_GENERATED"
            confidence = ensemble_prob_fake
        
        # Create probability dict
        prob_dict = {
            "HUMAN": round(ensemble_prob_real, 4),
            "AI_GENERATED": round(ensemble_prob_fake, 4)
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
                "ast_prob_real": round(calibrated_ast_real, 4),
                "acoustic_score": round(acoustic_real_score, 4),
                "ensemble_method": "50% Calibrated AST + 50% Enhanced Acoustic + Bias Correction",
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
    
    def _lightweight_acoustic_analysis(self, audio: np.ndarray) -> float:
        """
        Enhanced acoustic feature analysis to detect human voice characteristics.
        Returns probability that audio is real human speech (0-1).
        """
        try:
            human_scores = []
            
            # Feature 1: Enhanced Spectral Centroid Analysis
            hop_length = 512
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=self.SAMPLE_RATE, hop_length=hop_length
            )[0]
            
            if len(spectral_centroids) > 1:
                # Human voices have more irregular spectral content
                sc_variance = np.var(spectral_centroids)
                sc_mean = np.mean(spectral_centroids)
                
                # Multiple indicators for spectral centroid
                if 1000 < sc_variance < 80000:  # Strong human indicators
                    human_scores.append(0.8)
                elif 500 < sc_variance < 100000:  # Moderate human indicators
                    human_scores.append(0.6)
                elif sc_variance > 100:  # Some variation is better than none
                    human_scores.append(0.4)
                else:
                    human_scores.append(0.2)  # Very uniform (AI-like)
                
                # Spectral centroid in human range
                if 500 < sc_mean < 4000:  # Typical speech range
                    human_scores.append(0.7)
                elif 200 < sc_mean < 6000:  # Extended speech range
                    human_scores.append(0.5)
                else:
                    human_scores.append(0.3)
            
            # Feature 2: Enhanced Energy Dynamics
            frame_length = 1024
            energy_frames = []
            for i in range(0, len(audio) - frame_length, frame_length//2):  # Overlapping frames
                frame = audio[i:i + frame_length]
                energy_frames.append(np.sum(frame ** 2))
            
            if len(energy_frames) > 3:
                energy_std = np.std(energy_frames)
                energy_mean = np.mean(energy_frames) + 1e-10
                energy_cv = energy_std / energy_mean
                
                # Multiple thresholds for energy variation
                if energy_cv > 1.0:  # High variation (very human-like)
                    human_scores.append(0.9)
                elif energy_cv > 0.5:  # Moderate variation
                    human_scores.append(0.7)
                elif energy_cv > 0.2:  # Some variation
                    human_scores.append(0.5)
                else:
                    human_scores.append(0.2)  # Very stable (AI-like)
            
            # Feature 3: Zero-Crossing Rate Patterns (Enhanced)
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
            if len(zcr) > 1:
                zcr_mean = np.mean(zcr)
                zcr_std = np.std(zcr)
                
                # Human speech has specific ZCR characteristics
                if 0.05 < zcr_mean < 0.25 and zcr_std > 0.02:  # Ideal human range
                    human_scores.append(0.8)
                elif 0.02 < zcr_mean < 0.35 and zcr_std > 0.01:  # Extended human range
                    human_scores.append(0.6)
                elif zcr_std > 0.005:  # Some variation is good
                    human_scores.append(0.4)
                else:
                    human_scores.append(0.2)
            
            # Feature 4: Formant-like Analysis (Enhanced)
            stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop_length))
            freq_bins = librosa.fft_frequencies(sr=self.SAMPLE_RATE, n_fft=2048)
            
            # Human formant regions
            f1_indices = np.where((freq_bins >= 200) & (freq_bins <= 1200))[0]
            f2_indices = np.where((freq_bins >= 800) & (freq_bins <= 3500))[0]
            f3_indices = np.where((freq_bins >= 2000) & (freq_bins <= 4500))[0]
            
            formant_scores = []
            if len(f1_indices) > 0:
                f1_energy = np.mean(stft[f1_indices, :])
                if f1_energy > 0.01:  # Significant energy in F1 region
                    formant_scores.append(0.8)
            
            if len(f2_indices) > 0:
                f2_energy = np.mean(stft[f2_indices, :])
                if f2_energy > 0.005:  # Energy in F2 region
                    formant_scores.append(0.7)
            
            if len(f3_indices) > 0:
                f3_energy = np.mean(stft[f3_indices, :])
                if f3_energy > 0.003:  # Energy in F3 region
                    formant_scores.append(0.6)
            
            if formant_scores:
                human_scores.append(np.mean(formant_scores))
            else:
                human_scores.append(0.3)
            
            # Feature 5: Pitch Tracking (Enhanced)
            try:
                # Use harmonic-percussive separation for better pitch detection
                y_harmonic = librosa.effects.harmonic(audio)
                pitches, magnitudes = librosa.core.piptrack(
                    y=y_harmonic, sr=self.SAMPLE_RATE, threshold=0.1
                )
                
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if len(pitch_values) > 5:
                    pitch_mean = np.mean(pitch_values)
                    pitch_std = np.std(pitch_values)
                    
                    # Human fundamental frequency analysis
                    if 80 < pitch_mean < 400:  # Core human F0 range
                        if 5 < pitch_std < 60:  # Natural variation
                            human_scores.append(0.9)
                        elif pitch_std > 2:  # Some variation
                            human_scores.append(0.7)
                        else:
                            human_scores.append(0.4)
                    elif 60 < pitch_mean < 500:  # Extended range
                        if pitch_std > 3:
                            human_scores.append(0.6)
                        else:
                            human_scores.append(0.4)
                    else:
                        human_scores.append(0.3)
                else:
                    human_scores.append(0.4)  # Minimal pitch info
            except:
                human_scores.append(0.5)  # Neutral if pitch analysis fails
            
            # Feature 6: Amplitude Irregularity (Enhanced)
            if len(audio) > 1600:  # Ensure enough samples
                # Analyze amplitude patterns in overlapping windows
                window_size = len(audio) // 20  # 20 windows
                amp_variations = []
                
                for i in range(15):  # 15 overlapping windows
                    start = i * window_size // 2
                    end = start + window_size
                    if end < len(audio):
                        window = audio[start:end]
                        amp_variations.append(np.std(window))
                
                if len(amp_variations) > 5:
                    amp_irregularity = np.std(amp_variations) / (np.mean(amp_variations) + 1e-10)
                    
                    if amp_irregularity > 0.8:  # High irregularity (human-like)
                        human_scores.append(0.8)
                    elif amp_irregularity > 0.4:  # Moderate irregularity
                        human_scores.append(0.6)
                    elif amp_irregularity > 0.1:  # Some irregularity
                        human_scores.append(0.4)
                    else:
                        human_scores.append(0.2)  # Very regular (AI-like)
                else:
                    human_scores.append(0.5)
            
            # Calculate final human probability with weighted averaging
            if human_scores:
                # Weight recent features more heavily as they're more sophisticated
                weights = [1.0, 1.2, 1.0, 1.5, 1.8, 1.3][:len(human_scores)]
                if len(weights) < len(human_scores):
                    weights.extend([1.0] * (len(human_scores) - len(weights)))
                
                weighted_score = np.average(human_scores, weights=weights)
                
                # Apply calibrated sigmoid for better distribution
                # This helps distinguish between clearly human, uncertain, and clearly AI
                calibrated_score = 1.0 / (1.0 + np.exp(-6 * (weighted_score - 0.5)))
                return float(np.clip(calibrated_score, 0.1, 0.9))  # Avoid extreme values
            else:
                return 0.5  # Neutral if no features
                
        except Exception as e:
            logger.warning(f"Enhanced acoustic analysis failed: {e}")
            return 0.5
    
    def _analyze_acoustic_features(self, audio: np.ndarray) -> float:
        """
        Analyze acoustic features to detect human voice characteristics.
        
        Human voices typically have:
        - Variable pitch (F0) with natural fluctuations
        - Higher spectral irregularity
        - More dynamic energy variations
        - Natural breathing patterns and micro-pauses
        
        Returns:
            Score between 0 (AI-like) and 1 (human-like)
        """
        scores = []
        
        # 1. Pitch variability analysis using zero-crossing rate variance
        frame_length = int(0.025 * self.SAMPLE_RATE)  # 25ms frames
        hop_length = int(0.010 * self.SAMPLE_RATE)    # 10ms hop
        n_frames = max(1, 1 + (len(audio) - frame_length) // hop_length)
        
        if n_frames > 1:
            # Zero-crossing rate variance (human voice has more variation)
            zcr_values = []
            for i in range(n_frames):
                start = i * hop_length
                end = min(start + frame_length, len(audio))
                frame = audio[start:end]
                if len(frame) > 1:
                    zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
                    zcr_values.append(zcr)
            
            if len(zcr_values) > 1:
                zcr_variance = np.var(zcr_values)
                # Higher variance = more human-like
                zcr_score = min(zcr_variance / 0.005, 1.0)
                scores.append(zcr_score)
        
        # 2. Energy dynamics (human voice has more dynamic range)
        if n_frames > 1:
            energies = []
            for i in range(n_frames):
                start = i * hop_length
                end = min(start + frame_length, len(audio))
                frame = audio[start:end]
                energies.append(np.sum(frame ** 2) + 1e-10)
            
            if len(energies) > 1:
                energy_range = np.max(energies) / (np.min(energies) + 1e-10)
                # Higher dynamic range = more human-like
                energy_score = min(np.log10(energy_range + 1) / 3, 1.0)
                scores.append(energy_score)
        
        # 3. Spectral centroid variance (human voice has varying brightness)
        try:
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=self.SAMPLE_RATE, n_fft=frame_length, hop_length=hop_length
            )[0]
            if len(spectral_centroids) > 1:
                sc_variance = np.var(spectral_centroids) / (np.mean(spectral_centroids) + 1e-10)
                # Higher variance = more human-like
                sc_score = min(sc_variance / 0.1, 1.0)
                scores.append(sc_score)
        except Exception:
            pass
        
        # 4. Spectral flatness (human voice has harmonic structure, lower flatness)
        try:
            spectral_flatness = librosa.feature.spectral_flatness(
                y=audio, n_fft=frame_length, hop_length=hop_length
            )[0]
            if len(spectral_flatness) > 0:
                mean_flatness = np.mean(spectral_flatness)
                # Lower flatness (more harmonic) = more human-like
                flatness_score = 1.0 - min(mean_flatness * 5, 1.0)
                scores.append(flatness_score)
        except Exception:
            pass
        
        # 5. Temporal irregularity (check for micro-variations)
        if len(audio) > 1000:
            # Calculate short-term variations
            chunk_size = len(audio) // 20
            chunk_energies = []
            for i in range(20):
                chunk = audio[i*chunk_size:(i+1)*chunk_size]
                chunk_energies.append(np.std(chunk))
            
            irregularity = np.std(chunk_energies) / (np.mean(chunk_energies) + 1e-10)
            # Higher irregularity = more human-like
            irregularity_score = min(irregularity * 2, 1.0)
            scores.append(irregularity_score)
        
        # Combine scores with equal weighting
        if scores:
            final_score = np.mean(scores)
        else:
            final_score = 0.5  # Neutral if no features computed
        
        return float(final_score)
    
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
