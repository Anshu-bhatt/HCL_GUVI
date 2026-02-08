# analyze_audio_details.py
"""
Deep analysis of audio files to understand their characteristics
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

def analyze_audio_file(file_path):
    """Analyze audio file in detail"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(file_path).name}")
    print('='*80)
    
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=None)
        
        print(f"\n📊 Basic Info:")
        print(f"  - Duration: {len(audio)/sr:.2f}s")
        print(f"  - Sample Rate: {sr}Hz")
        print(f"  - Samples: {len(audio)}")
        print(f"  - Channels: {'Mono' if audio.ndim == 1 else 'Stereo'}")
        
        # Audio characteristics
        print(f"\n🎵 Audio Characteristics:")
        print(f"  - Mean amplitude: {np.mean(np.abs(audio)):.6f}")
        print(f"  - Max amplitude: {np.max(np.abs(audio)):.6f}")
        print(f"  - RMS energy: {np.sqrt(np.mean(audio**2)):.6f}")
        print(f"  - Zero crossing rate: {np.mean(librosa.zero_crossings(audio)):.6f}")
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        print(f"\n🔊 Spectral Features:")
        print(f"  - Spectral centroid (mean): {np.mean(spectral_centroids):.2f} Hz")
        print(f"  - Spectral centroid (std): {np.std(spectral_centroids):.2f} Hz")
        print(f"  - Spectral rolloff (mean): {np.mean(spectral_rolloff):.2f} Hz")
        print(f"  - Spectral rolloff (std): {np.std(spectral_rolloff):.2f} Hz")
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        print(f"\n🎼 MFCC Analysis:")
        print(f"  - MFCC mean: {np.mean(mfccs):.6f}")
        print(f"  - MFCC std: {np.std(mfccs):.6f}")
        print(f"  - MFCC variance across time: {np.mean(np.var(mfccs, axis=1)):.6f}")
        
        # Detect if audio seems natural or synthetic
        # Natural speech typically has more variation
        spectral_variance = np.std(spectral_centroids)
        energy_variance = np.std(librosa.feature.rms(y=audio)[0])
        
        print(f"\n🔍 Naturalness Indicators:")
        print(f"  - Spectral variance: {spectral_variance:.2f} (higher = more natural)")
        print(f"  - Energy variance: {energy_variance:.6f} (higher = more natural)")
        
        # Check for pitch consistency (AI voices tend to be more consistent)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            pitch_std = np.std(pitch_values)
            print(f"  - Pitch variance: {pitch_std:.2f} Hz (lower = more synthetic)")
        
        return {
            'duration': len(audio)/sr,
            'spectral_variance': float(spectral_variance),
            'energy_variance': float(energy_variance),
            'mfcc_std': float(np.std(mfccs))
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DETAILED AUDIO ANALYSIS")
    print("="*80)
    
    files = [
        r"D:\development\workspace\HCL\HCL_GUVI\test_samples\data_voices_martin_voyels_exports_mono__-a-c3-2.wav",
        r"D:\development\workspace\HCL\HCL_GUVI\test_samples\data_voices_martin_voyels_exports_mono__-ou-c3.wav",
        r"D:\development\workspace\HCL\HCL_GUVI\test_samples\sample voice 1.mp3",
        r"D:\development\workspace\HCL\HCL_GUVI\test_samples\Namami_30s.mp3",
    ]
    
    results = {}
    for file in files:
        result = analyze_audio_file(file)
        if result:
            results[Path(file).name] = result
    
    # Compare
    print(f"\n\n{'='*80}")
    print("COMPARISON SUMMARY")
    print('='*80)
    
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Duration: {data['duration']:.2f}s")
        print(f"  Spectral variance: {data['spectral_variance']:.2f}")
        print(f"  Energy variance: {data['energy_variance']:.6f}")
        print(f"  MFCC std: {data['mfcc_std']:.6f}")
