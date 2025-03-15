# FastRTC TurnSense

A turn-taking detection system for conversational AI built on top of FastRTC. This package provides a custom implementation of FastRTC's ReplyOnPause that uses the TurnSense model ([latishab/turnsense](https://huggingface.co/latishab/turnsense)) to detect natural turn-taking in conversations.

## About TurnSense Model

This package uses the TurnSense model, an end-of-utterance (EOU) detection model specifically designed for real-time voice AI applications. Built on SmolLM2-135M and optimized for low-power devices, it offers high accuracy while maintaining efficient performance. The model is available in two versions:
- Standard model: 97.50% accuracy
- Quantized model: 93.75% accuracy (recommended for edge devices)

## Features

- Natural turn detection using linguistic and contextual cues
- Built on top of FastRTC's ReplyOnPause system
- Uses ONNX-optimized TurnSense model for efficient inference
- Supports real-time audio processing
- Customizable turn detection thresholds

## Installation

```bash
pip install fastrtc-turnsense
```

## Quick Start

Here's an example of creating a voice chat application with AI:

```python
import os
from fastapi import FastAPI
import gradio as gr
from fastrtc import Stream, get_tts_model
from fastrtc_turnsense import TurnDetector, TurnDetectorOptions
from fastrtc.pause_detection.silero import SileroVadOptions
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize TTS model
tts_model = get_tts_model()

def echo(text_input):
    # Get AI response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text_input}],
        max_tokens=200,
    )
    response_text = response.choices[0].message.content
    
    # Convert to speech
    yield tts_model.tts(response_text)

# Configure turn detection
turn_options = TurnDetectorOptions(
    model_name="model_quantized.onnx",  # or "model_preprocessed.onnx" for non-quantized version
    audio_chunk_duration=3.0,
    turn_end_threshold=0.5,
)

# Configure VAD options
vad_options = SileroVadOptions(
    threshold=0.4,
    min_speech_duration_ms=250,
    min_silence_duration_ms=300,
    speech_pad_ms=200
)

# Create FastAPI app
app = FastAPI()

# Create stream with turn detector
stream = Stream(
    handler=TurnDetector(
        echo,
        algo_options=turn_options,
        input_sample_rate=16000,
        model_options=vad_options,
    ),
    modality="audio",
    mode="send-receive",
    ui_args={"title": "Voice Chat with AI"}
)

# Mount Gradio UI
app = gr.mount_gradio_app(app, stream.ui, path="/")

if __name__ == "__main__":
    stream.ui.launch()
```

## Environment Setup

Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_key_here
```

## Dependencies

Core dependencies:
- fastrtc
- numpy
- onnxruntime
- transformers
- huggingface-hub
- librosa

For the voice chat example, additional dependencies:
- gradio
- fastapi
- openai
- python-dotenv

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package or the TurnSense model in your research, please cite:

```bibtex
@software{latishab2025turnsense,
  author       = {Latisha Besariani HENDRA},
  title        = {TurnSense: A Lightweight End-of-Utterance Detection Model},
  month        = mar,
  year         = 2025,
  publisher    = {GitHub},
  journal      = {GitHub repository},
  url          = {https://github.com/latishab/turnsense},
  note         = {https://huggingface.co/latishab/turnsense}
}
```
