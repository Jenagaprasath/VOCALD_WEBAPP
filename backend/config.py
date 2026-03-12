"""
Vocald Performance Configuration
Adjust these settings based on your needs
"""

# PERFORMANCE MODE
# Options: 'speed', 'balanced', 'accuracy'
PERFORMANCE_MODE = 'speed'

# MODE CONFIGURATIONS
CONFIGS = {
    'speed': {
        'max_speakers': 10,              # Lower = faster diarization
        'max_segments_per_speaker': 3,   # Fewer segments = faster processing
        'min_segment_duration': 0.2,     # Higher = skip more short segments
        'segment_merge_gap': 0.3,        # Merge close segments
        'parallel_workers': 4,           # Number of parallel threads
        'use_simple_similarity': True,   # Use only cosine similarity
        'skip_outlier_removal': True,    # Skip outlier detection for speed
        'audio_preprocessing': 'minimal',# Skip heavy audio filters
    },
    'balanced': {
        'max_speakers': 15,
        'max_segments_per_speaker': 5,
        'min_segment_duration': 0.15,
        'segment_merge_gap': 0.25,
        'parallel_workers': 3,
        'use_simple_similarity': False,  # Use multi-metric similarity
        'skip_outlier_removal': False,
        'audio_preprocessing': 'standard',
    },
    'accuracy': {
        'max_speakers': 20,
        'max_segments_per_speaker': 7,
        'min_segment_duration': 0.15,
        'segment_merge_gap': 0.25,
        'parallel_workers': 2,
        'use_simple_similarity': False,
        'skip_outlier_removal': False,
        'audio_preprocessing': 'full',
    }
}

# Get active configuration
ACTIVE_CONFIG = CONFIGS[PERFORMANCE_MODE]

# SIMILARITY THRESHOLD
SIMILARITY_THRESHOLD = 0.65  # Lower = more lenient matching

# AI DETECTION
AI_DETECTION_THRESHOLD = 0.01  # Variance threshold for AI detection

# DATABASE
DB_TIMEOUT = 10.0  # seconds

# LOGGING
SHOW_DETAILED_LOGS = False  # Set to True for verbose output

print(f"⚙️ Vocald Performance Mode: {PERFORMANCE_MODE.upper()}")
print(f"   - Max speakers: {ACTIVE_CONFIG['max_speakers']}")
print(f"   - Segments per speaker: {ACTIVE_CONFIG['max_segments_per_speaker']}")
print(f"   - Parallel workers: {ACTIVE_CONFIG['parallel_workers']}")
print(f"   - Audio preprocessing: {ACTIVE_CONFIG['audio_preprocessing']}")
