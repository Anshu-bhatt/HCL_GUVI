# test_all_samples.py
"""
Test all audio samples in test_samples folder using the API
"""

import requests
import base64
import os
import json
import time
from pathlib import Path

API_URL = "http://localhost:8000"
SAMPLES_DIR = r"D:\development\workspace\HCL\HCL_GUVI\test_samples"

def convert_audio_to_base64(audio_file_path):
    """Convert audio file to Base64 string"""
    try:
        with open(audio_file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_base64
    except Exception as e:
        print(f"❌ Error converting {audio_file_path}: {e}")
        return None

def test_server_connection():
    """Check if server is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def test_audio_sample(file_path, language="Tamil"):
    """Test a single audio sample"""
    print(f"\n{'='*80}")
    print(f"Testing: {file_path}")
    print('='*80)
    
    # Convert to base64
    audio_base64 = convert_audio_to_base64(file_path)
    if audio_base64 is None:
        return None
    
    file_size = os.path.getsize(file_path)
    print(f"✓ File size: {file_size:,} bytes")
    print(f"✓ Base64 length: {len(audio_base64):,} characters")
    
    # Determine audio format from extension
    file_ext = Path(file_path).suffix.lower()
    audio_format = file_ext[1:] if file_ext else "mp3"
    
    print(f"✓ Format: {audio_format}")
    print(f"✓ Language: {language}")
    
    # Send detection request
    print("\n⏳ Sending detection request...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            json={
                "language": language,
                "audioFormat": audio_format,
                "audioBase64": audio_base64
            },
            timeout=120  # 2 minutes timeout for long audio
        )
        
        request_time = time.time() - start_time
        print(f"✓ Response received in {request_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n{'📊 DETECTION RESULTS':^80}")
            print('='*80)
            print(f"Status: {data.get('status', 'unknown')}")
            
            if data.get('status') == 'success':
                classification = data.get('classification', 'UNKNOWN')
                confidence = data.get('confidenceScore', 0.0)
                confidence_level = data.get('confidenceLevel', 'UNKNOWN')
                
                print(f"Classification: {classification}")
                print(f"Confidence Score: {confidence:.2%}")
                print(f"Confidence Level: {confidence_level}")
                print(f"Language: {data.get('language', 'unknown')}")
                
                if data.get('explanation'):
                    print(f"\nExplanation:")
                    print(f"  {data['explanation']}")
                
                if data.get('details'):
                    print(f"\nDetails:")
                    for key, value in data['details'].items():
                        if isinstance(value, float):
                            print(f"  - {key}: {value:.4f}")
                        else:
                            print(f"  - {key}: {value}")
                
                # Highlight the result
                if classification == 'AI_GENERATED':
                    print(f"\n{'⚠️  AI-GENERATED VOICE DETECTED':^80}")
                elif classification == 'HUMAN':
                    print(f"\n{'✅ HUMAN VOICE DETECTED':^80}")
                else:
                    print(f"\n{'❓ UNCERTAIN CLASSIFICATION':^80}")
            else:
                print(f"Message: {data.get('message', 'Unknown error')}")
                if data.get('details'):
                    print(f"Details: {data['details']}")
            
            print('='*80)
            
            return data
        else:
            print(f"❌ Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after {time.time() - start_time:.2f}s")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("AI VOICE DETECTION - BATCH TEST FOR ALL SAMPLES")
    print("="*80)
    
    # Check server connection
    print("\nChecking server connection...")
    if not test_server_connection():
        print(f"❌ Server not running at {API_URL}")
        print("\n⚠️  Please start the server first:")
        print("   python main.py")
        return
    
    print(f"✓ Server is running at {API_URL}")
    
    # Get all audio files from test_samples folder
    samples_path = Path(SAMPLES_DIR)
    if not samples_path.exists():
        print(f"❌ Folder not found: {SAMPLES_DIR}")
        return
    
    # Find all audio files (excluding .base64.txt files)
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(samples_path.glob(f'*{ext}')))
    
    if not audio_files:
        print(f"❌ No audio files found in {SAMPLES_DIR}")
        return
    
    print(f"\n✓ Found {len(audio_files)} audio file(s) to test")
    
    # Test each audio file
    results = {}
    
    for idx, audio_file in enumerate(sorted(audio_files), 1):
        print(f"\n[{idx}/{len(audio_files)}]")
        
        # Auto-detect language based on filename
        filename = audio_file.name.lower()
        if 'namami' in filename or 'tamil' in filename:
            language = "Tamil"
        elif 'hindi' in filename:
            language = "Hindi"
        elif 'english' in filename or 'martin' in filename:
            language = "English"
        elif 'malayalam' in filename:
            language = "Malayalam"
        elif 'telugu' in filename:
            language = "Telugu"
        else:
            language = "Tamil"  # Default
        
        result = test_audio_sample(str(audio_file), language)
        results[audio_file.name] = result
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ALL TESTS")
    print('='*80 + "\n")
    
    human_count = 0
    ai_count = 0
    uncertain_count = 0
    failed_count = 0
    
    for filename, result in results.items():
        if result is None or result.get('status') == 'error':
            error_msg = result.get('message', 'Unknown error') if result else 'Failed to process'
            status = f"❌ ERROR: {error_msg}"
            failed_count += 1
        elif result.get('classification') == 'HUMAN':
            confidence = result.get('confidenceScore', 0)
            status = f"✅ HUMAN ({confidence:.1%} - {result.get('confidenceLevel', 'N/A')})"
            human_count += 1
        elif result.get('classification') == 'AI_GENERATED':
            confidence = result.get('confidenceScore', 0)
            status = f"⚠️  AI ({confidence:.1%} - {result.get('confidenceLevel', 'N/A')})"
            ai_count += 1
        else:
            confidence = result.get('confidenceScore', 0)
            status = f"❓ UNCERTAIN ({confidence:.1%})"
            uncertain_count += 1
        
        print(f"{filename:50s} → {status}")
    
    print(f"\n{'='*80}")
    print(f"Total Tested: {len(audio_files)}")
    print(f"Human Voices: {human_count}")
    print(f"AI Voices: {ai_count}")
    print(f"Uncertain: {uncertain_count}")
    print(f"Failed: {failed_count}")
    print('='*80 + "\n")
    
    # Save detailed results to JSON
    results_file = "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
