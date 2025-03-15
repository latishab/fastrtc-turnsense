"""
TurnSense: Turn-taking detection system for conversational AI.

This package provides a custom implementation of FastRTC's ReplyOnPause
that uses a turnsense model to detect natural turn-taking in conversations.
"""

from .__about__ import (
    __version__,
    __author__,
    __author_email__,
    __license__,
    __copyright__,
)

from typing import Callable, Literal
import logging
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
import librosa
import asyncio
from typing import cast
from time import time
from difflib import SequenceMatcher
from typing import List, Tuple

from fastrtc.reply_on_pause import (
    AlgoOptions,
    AppState,
    ModelOptions,
    PauseDetectionModel,
    ReplyFnGenerator,
    ReplyOnPause,
    create_message,
    split_output,
)
from fastrtc.speech_to_text import get_stt_model, stt_for_chunks
from fastrtc.utils import audio_to_float32

class ColorFormatter(logging.Formatter):
    green = "\x1b[32;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def format(self, record):
        color = self.green if record.levelno == logging.INFO else self.grey
        record.levelname = f"{color}{record.levelname}{self.reset}"
        return super().format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter('%(levelname)s: %(message)s'))
logger.handlers = [console_handler]


@dataclass
class TurnDetectorOptions(AlgoOptions):
    """Configuration options for TurnDetector."""
    model_name: str = "model_quantized.onnx"
    turn_end_threshold: float = 0.5
    audio_chunk_duration: float = 5.0


class TurnDetector(ReplyOnPause):
    """Turn detection using linguistic and contextual cues."""
    
    def __init__(
        self,
        fn: ReplyFnGenerator,
        startup_fn: Callable | None = None,
        algo_options: TurnDetectorOptions | None = None,
        model_options: ModelOptions | None = None,
        can_interrupt: bool = True,
        expected_layout: Literal["mono", "stereo"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int = 480,
        input_sample_rate: int = 48000,
        model: PauseDetectionModel | None = None,
    ):
        self.turn_options = algo_options or TurnDetectorOptions()
        self.window_duration = 5.0
        
        super().__init__(
            fn,
            algo_options=self.turn_options,
            startup_fn=startup_fn,
            model_options=model_options,
            can_interrupt=can_interrupt,
            expected_layout=expected_layout,
            output_sample_rate=output_sample_rate,
            output_frame_size=output_frame_size,
            input_sample_rate=input_sample_rate,
            model=model,
        )
        
        self.stt_model = get_stt_model("moonshine/base")
        logger.info(f"Initializing turn detector with model: {self.turn_options.model_name}")

        self.model_path = hf_hub_download(
            repo_id="latishab/turnsense",
            filename=self.turn_options.model_name,
            local_dir="models"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("latishab/turnsense")
        self.session = None
        self.warmed_up = False
        self.warmup()
        self._initialize_state()

    def _initialize_state(self):
        """Initialize or reset state variables."""
        self.state = AppState()
        self.state.stream = None
        self.state.sampling_rate = self.input_sample_rate
        self.state.last_text = ""

    def warmup(self) -> None:
        """Initialize the turn detection model."""
        if not self.warmed_up:
            logger.info("Warming up turn detection model...")
            self.session = ort.InferenceSession(self.model_path)
            dummy_text = "<|user|> Hello, how are you? <|im_end|>"
            inputs = self.tokenizer(
                dummy_text,
                padding="max_length",
                max_length=256,
                return_tensors="np"
            )
            self.session.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
            )
            self.warmed_up = True
            logger.info("Turn detection model warmed up successfully")

    def check_turn_end(self, text: str) -> float:
        """Return probability that current text marks end of turn."""
        if not self.warmed_up:
            self.warmup()
            
        formatted_text = f"<|user|> {text} <|im_end|>"
        inputs = self.tokenizer(
            formatted_text,
            padding="max_length",
            max_length=256,
            return_tensors="np"
        )
        
        output = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
        )[0]
        
        if len(output.shape) > 1 and output.shape[1] == 2:
            prob = float(output[0, 1])  
        else:
            prob = float(output[0]) 
            
        return prob

    def determine_pause(self, audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
        """Process audio and determine if turn has ended."""
        duration = len(audio) / sampling_rate
        if duration >= self.turn_options.audio_chunk_duration:
            # Get text from STT
            text = self.stt_model.stt((sampling_rate, audio))
            if not text or not (text := text.strip()):
                return False

            # Check for turn end
            turn_end_prob = self.check_turn_end(text)
            if turn_end_prob >= self.turn_options.turn_end_threshold:
                state.last_text = text
                state.stream = audio.copy()
                logger.info(f"Turn end: '{state.last_text}'")
                return True

            # Update state with current transcription
            state.last_text = text
            logger.info(f"Processing: '{text}'")

        return False

    def reset(self):
        """Reset all state variables."""
        super().reset()
        self.generator = None
        self.event.clear()
        self._initialize_state()

    def copy(self):
        """Create a copy of this handler."""
        return TurnDetector(
            self.fn,
            self.startup_fn,
            self.turn_options,
            self.model_options,
            self.can_interrupt,
            self.expected_layout,
            self.output_sample_rate,
            self.output_frame_size,
            self.input_sample_rate,
            self.model,
        )

    def _preserve_state(self, new_state: AppState):
        """Preserve custom state fields when state is recreated."""
        new_state.last_text = getattr(self.state, 'last_text', '')
        new_state.sampling_rate = self.input_sample_rate
        return new_state

    def emit(self):
        """Override emit to pass only the text."""
        if not self.event.is_set():
            return None
        else:
            if not self.generator:
                self.send_message_sync(create_message("log", "pause_detected"))
                if self._needs_additional_inputs and not self.args_set.is_set():
                    if not self.phone_mode:
                        self.wait_for_args_sync()
                    else:
                        self.latest_args = [None]
                        self.args_set.set()
                text = getattr(self.state, 'last_text', '')
                if self._needs_additional_inputs:
                    self.latest_args[0] = text
                    self.generator = self.fn(*self.latest_args)
                else:
                    self.generator = self.fn(text)
                self.state = self._preserve_state(self.state.new())
            self.state.responding = True
            try:
                if self.is_async:
                    output = asyncio.run_coroutine_threadsafe(
                        self.async_iterate(self.generator), self.loop
                    ).result()
                else:
                    output = next(self.generator)
                audio, additional_outputs = split_output(output)
                if audio is not None:
                    self.send_message_sync(create_message("log", "response_starting"))
                    self.state.responded_audio = True
                if self.phone_mode:
                    if additional_outputs:
                        self.latest_args = [None] + list(additional_outputs.args)
                return output
            except (StopIteration, StopAsyncIteration):
                if not self.state.responded_audio:
                    self.send_message_sync(create_message("log", "response_starting"))
                self.reset()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.reset()
                raise e

__all__ = ["TurnDetector", "TurnDetectorOptions"]