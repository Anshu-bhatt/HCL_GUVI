#!/usr/bin/env python3
"""
Simple test to verify the bias correction is working
"""

import requests
import os

def test_api():
    # Start with a simple health check
    try:
        response = requests.get('http://localhost:8000/')
        print(f"✅ API Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ API not running: {e}")
        return False
    
    # Test with a known human file  
    test_file = "test_samples/linus-original-DEMO.mp3"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        # Try alternative locations
        alternatives = [
            "test_sample/tutorial-video.mp3",
            "test_sample/human-vowel-sounds.wav"
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                test_file = alt
                print(f"📁 Using alternative: {test_file}")
                break
        else:
            print("❌ No test files found")
            return False
    
    print(f"\n🎵 Testing file: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (os.path.basename(test_file), f, 'audio/mpeg')}
            response = requests.post('http://localhost:8000/detect', files=files)
        
        if response.status_code == 200:
            result = response.json()
            classification = result['classification']
            confidence = result['confidence']
            
            print(f"🔍 Classification: {classification}")
            print(f"📊 Confidence: {confidence:.1%}")
            
            if 'details' in result:
                details = result['details']
                ast_score = details.get('ast_prob_real', 'N/A')
                acoustic_score = details.get('acoustic_score', 'N/A')
                method = details.get('ensemble_method', 'Unknown')
                
                print(f"🧠 AST Score (Real): {ast_score}")
                print(f"🎤 Acoustic Score: {acoustic_score}")
                print(f"⚙️  Method: {method}")
                
                # Check if bias correction worked
                if classification == 'HUMAN':
                    print("✅ SUCCESS: Model correctly identified human voice!")
                    print("🎯 Bias correction appears to be working")
                else:
                    print("❌ ISSUE: Still classifying human voice as AI")
                    print("🔧 May need further bias adjustment")
            
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Bias-Corrected VoiceGUARD Model")
    print("=" * 50)
    test_api()