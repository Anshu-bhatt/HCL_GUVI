# process_audio.py
"""
Process real audio files uploaded by the user.
Supports: WAV, MP3, FLAC, OGG, M4A formats
"""

import os
import sys
import argparse
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from audio_preprocessor import AudioProcessor
from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma']


def load_audio_file(file_path: str, sample_rate: int = 16000) -> tuple:
    """
    Load an audio file from disk.
    
    Args:
        file_path: Path to the audio file
        sample_rate: Target sample rate (Hz)
        
    Returns:
        Tuple of (audio_array, sample_rate)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check file extension
    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {ext}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Load audio using librosa
    logger.info(f"Loading audio file: {path.name}")
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    
    logger.info(f"✓ Loaded: {len(audio)/sr:.2f}s, {sr}Hz, {len(audio)} samples")
    return audio, sr


def process_audio_file(
    file_path: str,
    extract_features: bool = True,
    preprocess: bool = True,
    save_processed: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process an uploaded audio file.
    
    Args:
        file_path: Path to the audio file
        extract_features: Whether to extract audio features
        preprocess: Whether to preprocess the audio
        save_processed: Whether to save the processed audio
        output_dir: Directory to save processed audio (default: same as input)
        
    Returns:
        Dictionary containing processing results
    """
    results = {
        'file': file_path,
        'status': 'success',
        'info': {},
        'features': {},
        'processed_file': None
    }
    
    try:
        # Initialize processor
        processor = AudioProcessor(
            sample_rate=config.SAMPLE_RATE,
            max_duration=config.MAX_AUDIO_LENGTH_SECONDS
        )
        
        # Load the audio file
        audio, sr = load_audio_file(file_path, config.SAMPLE_RATE)
        
        # Get audio info
        results['info'] = processor.get_audio_info(audio)
        
        # Preprocess if requested
        if preprocess:
            logger.info("Preprocessing audio...")
            processed_audio = processor.preprocess_audio(audio)
            results['info']['preprocessed_length'] = len(processed_audio) / sr
        else:
            processed_audio = audio
        
        # Extract features if requested
        if extract_features:
            logger.info("Extracting features...")
            features = processor.extract_features(processed_audio)
            results['features'] = {k: float(v) for k, v in features.items()}
        
        # Save processed audio if requested
        if save_processed:
            import soundfile as sf
            
            if output_dir is None:
                output_dir = Path(file_path).parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename
            input_name = Path(file_path).stem
            output_path = output_dir / f"{input_name}_processed.wav"
            
            # Save as WAV
            sf.write(str(output_path), processed_audio, sr)
            results['processed_file'] = str(output_path)
            logger.info(f"✓ Saved processed audio: {output_path}")
        
        logger.info("✓ Audio processing complete!")
        
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        logger.error(f"Processing failed: {e}")
    
    return results


def process_directory(
    dir_path: str,
    extract_features: bool = True,
    preprocess: bool = True
) -> list:
    """
    Process all audio files in a directory.
    
    Args:
        dir_path: Path to directory containing audio files
        extract_features: Whether to extract features
        preprocess: Whether to preprocess audio
        
    Returns:
        List of processing results for each file
    """
    dir_path = Path(dir_path)
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")
    
    # Find all audio files
    audio_files = []
    for ext in SUPPORTED_FORMATS:
        audio_files.extend(dir_path.glob(f"*{ext}"))
        audio_files.extend(dir_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        logger.warning(f"No audio files found in {dir_path}")
        return []
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Process each file
    results = []
    for audio_file in audio_files:
        logger.info(f"\nProcessing: {audio_file.name}")
        result = process_audio_file(
            str(audio_file),
            extract_features=extract_features,
            preprocess=preprocess
        )
        results.append(result)
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Pretty print processing results"""
    print("\n" + "=" * 60)
    print("AUDIO PROCESSING RESULTS")
    print("=" * 60)
    
    print(f"\nFile: {results['file']}")
    print(f"Status: {results['status']}")
    
    if results['status'] == 'error':
        print(f"Error: {results.get('error', 'Unknown error')}")
        return
    
    print("\n--- Audio Info ---")
    for key, value in results['info'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    if results['features']:
        print("\n--- Extracted Features (Top 10) ---")
        # Sort by absolute value and show top 10
        sorted_features = sorted(
            results['features'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        for key, value in sorted_features:
            print(f"  {key}: {value:.4f}")
        print(f"  ... and {len(results['features']) - 10} more features")
    
    if results['processed_file']:
        print(f"\nProcessed file saved: {results['processed_file']}")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Process audio files for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_audio.py audio.mp3
  python process_audio.py audio.wav --no-features
  python process_audio.py audio.mp3 --save --output-dir ./processed
  python process_audio.py --directory ./audio_folder
        """
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='Path to audio file to process'
    )
    parser.add_argument(
        '--directory', '-d',
        help='Process all audio files in a directory'
    )
    parser.add_argument(
        '--no-features',
        action='store_true',
        help='Skip feature extraction'
    )
    parser.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Skip preprocessing'
    )
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save processed audio to file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for processed files'
    )
    
    args = parser.parse_args()
    
    # Check if input is provided
    if not args.file and not args.directory:
        parser.print_help()
        print("\n❌ Error: Please provide an audio file or directory")
        sys.exit(1)
    
    # Process directory
    if args.directory:
        results = process_directory(
            args.directory,
            extract_features=not args.no_features,
            preprocess=not args.no_preprocess
        )
        for result in results:
            print_results(result)
    
    # Process single file
    elif args.file:
        results = process_audio_file(
            args.file,
            extract_features=not args.no_features,
            preprocess=not args.no_preprocess,
            save_processed=args.save,
            output_dir=args.output_dir
        )
        print_results(results)


if __name__ == "__main__":
    main()
