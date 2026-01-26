# audio_processor.py

import librosa
import numpy as np
import base64
import io
import soundfile as sf
from typing import Tuple, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles all audio processing operations:
    - Base64 decoding
    - Audio loading and validation
    - Preprocessing and normalization
    - Feature extraction
    """
    
    def __init__(self, sample_rate: int = 16000, max_duration: int = 30):
        """
        Initialize AudioProcessor
        
        Args:
            sample_rate: Target sample rate for audio (Hz)
            max_duration: Maximum allowed audio duration (seconds)
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = sample_rate * max_duration
        
        logger.info(f"AudioProcessor initialized: SR={sample_rate}Hz, Max={max_duration}s")
    
    def decode_base64_audio(self, audio_base64: str) -> Tuple[np.ndarray, int]:
        """
        Decode Base64 encoded MP3 to audio array
        
        Args:
            audio_base64: Base64 encoded audio string
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            ValueError: If decoding or loading fails
        """
        try:
            # Step 1: Decode Base64 to bytes
            logger.info("Decoding Base64 audio...")
            audio_bytes = base64.b64decode(audio_base64)
            logger.info(f"Decoded {len(audio_bytes)} bytes")
            
            # Step 2: Create file-like object
            audio_buffer = io.BytesIO(audio_bytes)
            
            # Step 3: Load audio using librosa
            # librosa can handle MP3 if ffmpeg is installed
            audio, sr = librosa.load(
                audio_buffer,
                sr=self.sample_rate,  # Resample to target rate
                mono=True,             # Convert to mono
                dtype=np.float32
            )
            
            logger.info(f"Audio loaded: shape={audio.shape}, sr={sr}Hz")
            
            # Step 4: Validate audio
            self._validate_audio(audio)
            
            return audio, sr
            
        except base64.binascii.Error as e:
            logger.error(f"Base64 decoding failed: {e}")
            raise ValueError(f"Invalid Base64 encoding: {str(e)}")
        
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            raise ValueError(f"Failed to load audio: {str(e)}")
    
    def _validate_audio(self, audio: np.ndarray) -> None:
        """
        Validate audio array
        
        Args:
            audio: Audio array to validate
            
        Raises:
            ValueError: If audio is invalid
        """
        # Check if audio is empty
        if len(audio) == 0:
            raise ValueError("Audio is empty")
        
        # Check if audio is too long
        if len(audio) > self.max_samples:
            raise ValueError(
                f"Audio too long: {len(audio)/self.sample_rate:.1f}s "
                f"(max {self.max_duration}s)"
            )
        
        # Check if audio is too short (at least 0.5 seconds)
        min_samples = int(0.5 * self.sample_rate)
        if len(audio) < min_samples:
            raise ValueError(
                f"Audio too short: {len(audio)/self.sample_rate:.1f}s "
                f"(min 0.5s)"
            )
        
        # Check for valid values (no NaN or Inf)
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            raise ValueError("Audio contains invalid values (NaN or Inf)")
        
        logger.info(f"✓ Audio validated: {len(audio)/self.sample_rate:.2f}s duration")
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for model input
        
        Steps:
        1. Normalize amplitude
        2. Remove silence
        3. Ensure fixed length (pad or truncate)
        
        Args:
            audio: Raw audio array
            
        Returns:
            Preprocessed audio array
        """
        logger.info("Preprocessing audio...")
        
        # Step 1: Normalize amplitude to [-1, 1]
        audio = self._normalize_audio(audio)
        
        # Step 2: Trim silence from beginning and end
        audio = self._trim_silence(audio)
        
        # Step 3: Ensure fixed length (4 seconds for consistency)
        target_length = 4 * self.sample_rate
        audio = self._fix_length(audio, target_length)
        
        logger.info(f"✓ Audio preprocessed: shape={audio.shape}")
        return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / (max_val + 1e-8)
        return audio
    
    def _trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Remove silence from audio"""
        try:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return audio_trimmed
        except:
            # If trimming fails, return original
            return audio
    
    def _fix_length(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate audio to fixed length"""
        if len(audio) < target_length:
            # Pad with zeros
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            # Truncate
            audio = audio[:target_length]
        
        return audio
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract audio features for analysis
        
        Features extracted:
        1. Spectral features (centroid, rolloff, bandwidth)
        2. Zero crossing rate
        3. MFCC (Mel-frequency cepstral coefficients)
        4. RMS energy
        5. Spectral contrast
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary of features
        """
        logger.info("Extracting audio features...")
        
        features = {}
        
        try:
            # 1. Spectral Centroid - indicates "brightness" of sound
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, 
                sr=self.sample_rate
            )[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)
            
            # 2. Spectral Rolloff - frequency below which X% of energy is contained
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, 
                sr=self.sample_rate
            )[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # 3. Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, 
                sr=self.sample_rate
            )[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # 4. Zero Crossing Rate - voice vs noise indicator
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 5. MFCC - captures timbral features
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=13
            )
            # Store mean and std for each MFCC coefficient
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
                features[f'mfcc_{i}_std'] = np.std(mfcc[i])
            
            # 6. RMS Energy - loudness indicator
            rms = librosa.feature.rms(y=audio)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # 7. Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio, 
                sr=self.sample_rate
            )
            features['spectral_contrast_mean'] = np.mean(spectral_contrast)
            features['spectral_contrast_std'] = np.std(spectral_contrast)
            
            logger.info(f"✓ Extracted {len(features)} features")
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return empty features on failure
            features = {}
        
        return features
    
    def get_audio_info(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Get basic audio information
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with audio info
        """
        duration = len(audio) / self.sample_rate
        
        info = {
            'duration_seconds': round(duration, 2),
            'num_samples': len(audio),
            'sample_rate': self.sample_rate,
            'max_amplitude': float(np.max(np.abs(audio))),
            'mean_amplitude': float(np.mean(np.abs(audio))),
            'rms_amplitude': float(np.sqrt(np.mean(audio**2)))
        }
        
        return info