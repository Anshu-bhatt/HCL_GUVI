#!/usr/bin/env python3
"""
Direct test of the bias-corrected detector without API
"""

import sys
import os
sys.path.append('.')

import numpy as np
import librosa
from voiceguard_detector import VoiceGUARDDetector

def test_direct():
    """Test the detector directly without API overhead"""
    print("🔧 Testing Bias-Corrected VoiceGUARD Detector (Direct)")
    print("=" * 55)
    
    # Initialize detector
    print("📥 Loading model...")
    detector = VoiceGUARDDetector()
    detector.load_model()
    print("✅ Model loaded successfully")
    
    # Test files
    test_files = [
        ("test_samples/linus-original-DEMO.mp3", "human"),
        ("test_sample/human-vowel-sounds.wav", "human"),
        ("test_sample/elevenlabs-AI-audio.wav", "ai"),
    ]
    
    results = []
    
    for file_path, expected_type in test_files:
        if not os.path.exists(file_path):
            print(f"⏭️  Skipping {file_path} (not found)")
            continue
            
        print(f"\n🎵 Testing: {os.path.basename(file_path)} (expected: {expected_type})")
        
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            
            # Classify using our bias-corrected ensemble
            result = detector.classify(audio, sr)
            
            classification = result['classification']
            confidence = result['confidence']
            details = result['details']
            
            # Show results with color coding
            if classification == 'HUMAN':
                emoji = "🟢"
            else:
                emoji = "🔴"
                
            print(f"{emoji} Result: {classification} ({confidence:.1%})")
            print(f"   📊 AST Score: {details.get('ast_prob_real', 'N/A')}")
            print(f"   🎤 Acoustic Score: {details.get('acoustic_score', 'N/A')}")
            print(f"   ⚙️  Method: {details.get('ensemble_method', 'Unknown')}")
            
            # Check accuracy
            is_correct = (
                (expected_type == 'human' and classification == 'HUMAN') or
                (expected_type == 'ai' and classification == 'AI_GENERATED')
            )
            
            if is_correct:
                print("   ✅ CORRECT classification!")
            else:
                print(f"   ❌ WRONG - Expected {expected_type.upper()}")
                
            results.append((file_path, expected_type, classification, confidence, is_correct))
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")
    
    # Summary
    if results:
        print("\n" + "=" * 55)
        print("📈 BIAS CORRECTION TEST RESULTS")
        print("=" * 55)
        
        correct = sum(1 for r in results if r[4])
        total = len(results)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        print(f"🎯 Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        # Check if we're correctly identifying humans
        human_tests = [r for r in results if r[1] == 'human']
        human_correct = sum(1 for r in human_tests if r[4])
        
        if human_tests:
            human_accuracy = (human_correct / len(human_tests)) * 100
            print(f"🟢 Human Detection: {human_correct}/{len(human_tests)} correct ({human_accuracy:.1f}%)")
            
            if human_correct > 0:
                print("✅ SUCCESS: Bias correction is working!")
                print("   Model can now detect human voices correctly")
            else:
                print("❌ ISSUE: Still failing to detect human voices")
                print("   Model may need additional bias adjustment")
        
        ai_tests = [r for r in results if r[1] == 'ai']
        ai_correct = sum(1 for r in ai_tests if r[4])
        
        if ai_tests:
            ai_accuracy = (ai_correct / len(ai_tests)) * 100
            print(f"🔴 AI Detection: {ai_correct}/{len(ai_tests)} correct ({ai_accuracy:.1f}%)")
    
    print("\n🌐 To test via API, visit: http://localhost:8000/docs")

if __name__ == "__main__":
    test_direct()