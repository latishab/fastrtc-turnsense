"""Tests for the TurnDetector class."""

import numpy as np
import pytest
from fastrtc.reply_on_pause import AppState

from fastrtc_turnsense import TurnDetector


def dummy_generator():
    """Dummy generator for testing."""
    yield (24000, np.zeros((1, 24000), dtype=np.int16))


def test_turn_detector_initialization():
    """Test that TurnDetector initializes correctly."""
    detector = TurnDetector(dummy_generator)
    assert detector is not None
    assert detector.algo_options.audio_chunk_duration == 3.0


def test_turn_detector_determine_pause():
    """Test that determine_pause works correctly."""
    detector = TurnDetector(dummy_generator)
    state = AppState()
    
    # Test with empty audio (should return False)
    audio = np.zeros(1000, dtype=np.float32)
    assert not detector.determine_pause(audio, 16000, state)

    # Test with short audio (should return False)
    audio = np.zeros(16000, dtype=np.float32)  # 1 second
    assert not detector.determine_pause(audio, 16000, state)


def test_turn_detector_copy():
    """Test that copy creates a new instance with same parameters."""
    detector = TurnDetector(dummy_generator)
    copied = detector.copy()
    
    assert copied is not detector  # Different instances
    assert copied.fn == detector.fn
    assert copied.algo_options.audio_chunk_duration == detector.algo_options.audio_chunk_duration 