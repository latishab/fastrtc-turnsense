# FastRTC TurnSense

A turn-taking detection system for conversational AI built on top of FastRTC. This package provides a custom implementation of FastRTC's ReplyOnPause that uses a machine learning model to detect natural turn-taking in conversations.

## Features

- Natural turn detection using linguistic and contextual cues
- Built on top of FastRTC's ReplyOnPause system
- Uses ONNX-optimized models for efficient inference
- Supports real-time audio processing
- Customizable turn detection thresholds

## Installation

```bash
pip install fastrtc-turnsense
```

## Quick Start

```python
from fastrtc_turnsense import TurnDetector, TurnDetectorOptions

# Configure the turn detector
options = TurnDetectorOptions(
    turn_end_threshold=0.5,
    audio_chunk_duration=5.0
)

# Create a turn detector instance
def reply_fn(text: str):
    # Your reply logic here
    return "Response to: " + text

detector = TurnDetector(
    fn=reply_fn,
    algo_options=options
)
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
