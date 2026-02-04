#!/usr/bin/env python3
"""
Test script to verify bias correction improvements
"""

import requests
import os
import json

def test_file_classification(file_path, expected_type=None):
    """Test a single audio file through the API"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
            response = requests.post('http://localhost:8000/detect', files=files)
        
        if response.status_code == 200:
            result = response.json()
            classification = result['classification']
            confidence = result['confidence']
            confidence_level = result['confidence_level']
            
            # Color coding for results
            if classification == 'HUMAN':
                color = "🟢"
            else:
                color = "🔴"
            
            print(f"{color} {os.path.basename(file_path)}: {classification} ({confidence:.1%}) - {confidence_level}")
            
            # Show detailed breakdown
            if 'details' in result:
                details = result['details']
                ast_score = details.get('ast_prob_real', 'N/A')
                acoustic_score = details.get('acoustic_score', 'N/A')
                print(f"   📊 AST: {ast_score}, Acoustic: {acoustic_score}")
            
            # Check if expectation matches
            if expected_type:
                if (expected_type == 'human' and classification == 'HUMAN') or \
                   (expected_type == 'ai' and classification == 'AI_GENERATED'):
                    print(f"   ✅ Correct classification!")
                else:
                    print(f"   ❌ Expected {expected_type}, got {classification}")
            
            return result
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Test failed for {file_path}: {e}")
        return None

def main():
    """Test the bias-corrected model"""
    print("🔧 Testing Bias-Corrected VoiceGUARD Model")
    print("=" * 50)
    
    # Test files from previous API tests
    test_files = [
        # Known human samples (should be classified as HUMAN)
        ("test_samples/linus-original-DEMO.mp3", "human"),
        ("test_sample/tutorial-video.mp3", "human"),
        ("test_sample/human-vowel-sounds.wav", "human"),
        
        # Known AI samples (should be classified as AI_GENERATED)  
        ("test_sample/elevenlabs-AI-audio.wav", "ai"),
    ]
    
    results = []
    human_correct = 0
    ai_correct = 0
    total_human = 0
    total_ai = 0
    
    for file_path, expected in test_files:
        print(f"\n🎵 Testing: {os.path.basename(file_path)}")
        result = test_file_classification(file_path, expected)
        
        if result:
            results.append((file_path, result))
            classification = result['classification']
            
            if expected == 'human':
                total_human += 1
                if classification == 'HUMAN':
                    human_correct += 1
            elif expected == 'ai':
                total_ai += 1
                if classification == 'AI_GENERATED':
                    ai_correct += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 BIAS CORRECTION TEST RESULTS")
    print("=" * 50)
    
    if total_human > 0:
        human_accuracy = (human_correct / total_human) * 100
        print(f"🟢 Human Detection: {human_correct}/{total_human} correct ({human_accuracy:.1f}%)")
    
    if total_ai > 0:
        ai_accuracy = (ai_correct / total_ai) * 100
        print(f"🔴 AI Detection: {ai_correct}/{total_ai} correct ({ai_accuracy:.1f}%)")
    
    total_tests = total_human + total_ai
    total_correct = human_correct + ai_correct
    
    if total_tests > 0:
        overall_accuracy = (total_correct / total_tests) * 100
        print(f"🎯 Overall Accuracy: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
        
        if human_correct > 0 and ai_correct > 0:
            print("✅ Model shows balanced detection (no bias towards AI)")
        elif human_correct == 0 and total_human > 0:
            print("❌ Model still biased towards AI detection")
        elif ai_correct == 0 and total_ai > 0:
            print("⚠️  Model may be biased towards human detection")
    
    print("\n🔄 To test manually, visit: http://localhost:8000/docs")

if __name__ == "__main__":
    main()