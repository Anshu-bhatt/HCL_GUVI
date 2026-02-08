# debug_detection.py
"""
Debug the detection process step by step
"""

import requests
import base64
import json

def test_file_detailed(file_path, language="English"):
    """Test a file and show all internal scores"""
    
    print(f"\n{'='*80}")
    print(f"DEBUGGING: {file_path}")
    print('='*80)
    
    # Load and encode
    with open(file_path, 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Send request
    response = requests.post(
        'http://localhost:8000/api/voice-detection',
        json={
            'language': language,
            'audioFormat': 'wav' if file_path.endswith('.wav') else 'mp3',
            'audioBase64': audio_base64
        },
        timeout=60
    )
    
    result = response.json()
    
    if result.get('status') == 'success':
        print(f"\n📊 FINAL RESULT:")
        print(f"  Classification: {result['classification']}")
        print(f"  Confidence: {result['confidenceScore']:.4f}")
        print(f"  Level: {result['confidenceLevel']}")
        
        if 'details' in result:
            details = result['details']
            print(f"\n🔍 INTERNAL SCORES:")
            print(f"  Wav2Vec2 Score: {details.get('wav2vec2_score', 'N/A'):.4f}")
            print(f"  Acoustic Score: {details.get('acoustic_score', 'N/A'):.4f}")
            print(f"  Combined Score: {details.get('combined_score', 'N/A'):.4f}")
            
            # Show if duration-aware logic was used
            if 'audio_duration_seconds' in details:
                duration = details['audio_duration_seconds']
                is_short = duration < 2.0
                print(f"\n🕐 DURATION INFO:")
                print(f"  Duration: {duration:.2f}s")
                print(f"  Short audio (< 2s): {'YES - Using adjusted weights!' if is_short else 'No'}")
            
            print(f"\n⚙️ THRESHOLDS:")
            print(f"  >= 0.60: AI-GENERATED (HIGH)")
            print(f"  >= 0.55: AI-GENERATED (MEDIUM)")
            print(f"  0.45-0.55: UNCERTAIN (LOW)")
            print(f"  0.30-0.45: HUMAN (MEDIUM)")
            print(f"  <  0.30: HUMAN (HIGH)")
            
            combined = details.get('combined_score', 0)
            print(f"\n💡 DECISION LOGIC:")
            print(f"  Combined score {combined:.4f} falls in:")
            if combined >= 0.60:
                print(f"  ➜ AI-GENERATED (HIGH) range")
            elif combined >= 0.55:
                print(f"  ➜ AI-GENERATED (MEDIUM) range")
            elif combined >= 0.45:
                print(f"  ➜ UNCERTAIN range")
            elif combined >= 0.30:
                print(f"  ➜ HUMAN (MEDIUM) range")
            else:
                print(f"  ➜ HUMAN (HIGH) range")
    else:
        print(f"❌ Error: {result.get('message')}")
    
    return result

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DETECTION DEBUG - COMPARING CONTROVERSIAL FILES")
    print("="*80)
    
    files = [
        (r"D:\development\workspace\HCL\HCL_GUVI\test_samples\data_voices_martin_voyels_exports_mono__-a-c3-2.wav", "English"),
        (r"D:\development\workspace\HCL\HCL_GUVI\test_samples\data_voices_martin_voyels_exports_mono__-ou-c3.wav", "English"),
        (r"D:\development\workspace\HCL\HCL_GUVI\test_samples\sample voice 1.mp3", "Tamil"),
    ]
    
    results = {}
    for file_path, lang in files:
        result = test_file_detailed(file_path, lang)
        results[file_path] = result
    
    # Summary comparison
    print(f"\n\n{'='*80}")
    print("COMPARISON OF SCORES")
    print('='*80)
    
    for file_path, result in results.items():
        if result.get('status') == 'success':
            filename = file_path.split('\\')[-1]
            combined = result['details']['combined_score']
            wav2vec = result['details']['wav2vec2_score']
            acoustic = result['details']['acoustic_score']
            
            print(f"\n{filename}:")
            print(f"  Wav2Vec2: {wav2vec:.4f} (70% weight)")
            print(f"  Acoustic: {acoustic:.4f} (30% weight)")
            print(f"  Combined: {combined:.4f}")
            print(f"  → {result['classification']} ({result['confidenceLevel']})")
