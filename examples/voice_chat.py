"""
Example application using the turnsense package for voice chat with AI.

This example demonstrates how to use turnsense with FastAPI and Gradio to create
a voice chat application that detects turn-taking in conversations.

To run this example:
1. Install the required dependencies: pip install -r requirements.txt
2. Set up your environment variables in .env
3. Run: python examples/voice_chat.py
"""

import os
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastrtc import (
    Stream,
    get_tts_model,
)
from fastrtc_turnsense import TurnDetector, TurnDetectorOptions
from fastrtc.pause_detection.silero import SileroVadOptions
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.handlers = [console_handler]  

# Enable DEBUG logging for turndetector only
turn_detector_logger = logging.getLogger("fastrtc_turnsense")
turn_detector_logger.setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()

# Initialize OpenAI client with DeepInfra
client = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)

# Initialize models
tts_model = get_tts_model()

def text_to_speech(text):
    try:
        return tts_model.tts(text)
    except Exception as e:
        logger.error(f"Error in text_to_speech: {e}")
        return (24000, np.zeros((1, 24000), dtype=np.int16))

def echo(text_input):
    try:
        # Validate we have text
        if not text_input or text_input.isspace():
            logger.warning("Empty text received")
            return None
        
        # Print the user input
        print(f"User: {text_input}")
            
        # Print AI response
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": text_input}],
            max_tokens=200,
        )
        response_text = response.choices[0].message.content
        print(f"AI: {response_text}")
        
        # Convert response to audio
        audio_output = text_to_speech(response_text)
        if audio_output is None:
            logger.error("Failed to convert response to speech")
            return None
            
        yield audio_output
            
    except Exception as e:
        logger.error(f"Error in echo: {e}", exc_info=True)
        yield None

# Configure turn detection options
turn_options = TurnDetectorOptions(
    model_name="model_preprocessed.onnx" if not os.getenv("USE_QUANTIZED_MODEL", "true").lower() == "true" else "model_quantized.onnx",
    audio_chunk_duration=3.0,     
    turn_end_threshold=0.5,  
)

# Configure VAD options with lower threshold
vad_options = SileroVadOptions(
    threshold=0.4,  
    min_speech_duration_ms=250,
    min_silence_duration_ms=300,   
    window_size_samples=1024,
    speech_pad_ms=200            
)

# Create the FastAPI app first
app = FastAPI()

# Create the stream with our handler
stream = Stream(
    handler=TurnDetector(
        echo,
        algo_options=turn_options,
        input_sample_rate=16000,
        model_options=vad_options,  
    ),
    modality="audio",
    mode="send-receive",
    ui_args={
        "title": "Voice Chat with AI (Powered by DeepInfra ⚡️)"
    }
)

# Mount the Gradio UI to FastAPI
app = gr.mount_gradio_app(app, stream.ui, path="/")
if __name__ == "__main__":    
    try:
        stream.ui.launch(share=True)
    except Exception as e:
        logger.error(f"Error launching app: {e}")