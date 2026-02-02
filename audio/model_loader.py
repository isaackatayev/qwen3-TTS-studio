"""Model loading utilities for Qwen3-TTS - isolated from Gradio UI."""

import os
import shutil
import gc
from pathlib import Path
from collections import OrderedDict

import torch

# Model paths configuration
MODEL_PATHS = {
    "1.7B-CustomVoice": "./Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6B-CustomVoice": "./Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7B-Base": "./Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B-Base": "./Qwen3-TTS-12Hz-0.6B-Base",
}

MAX_MODELS_ON_MPS = 1

# Shared model cache
loaded_models: OrderedDict = OrderedDict()


def _mps_cleanup():
    """Force MPS memory cleanup."""
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _unload_model(model_name: str) -> None:
    """Unload a model from memory."""
    if model_name in loaded_models:
        del loaded_models[model_name]
        _mps_cleanup()
        print(f"Unloaded {model_name}")


def get_model(model_name: str):
    """
    Get or load a Qwen3-TTS model.
    
    Args:
        model_name: Name of the model to load (e.g., "1.7B-CustomVoice")
        
    Returns:
        Loaded Qwen3TTSModel instance
        
    Raises:
        ValueError: If model path doesn't exist
        RuntimeError: If model loading fails
    """
    if model_name in loaded_models:
        loaded_models.move_to_end(model_name)
        return loaded_models[model_name]
    
    while len(loaded_models) >= MAX_MODELS_ON_MPS:
        old_name, _ = next(iter(loaded_models.items()))
        _unload_model(old_name)
    
    print(f"Loading {model_name}...")
    from qwen_tts import Qwen3TTSModel
    
    model_path = MODEL_PATHS.get(model_name)
    if not model_path:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_PATHS.keys())}")
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found: {model_path}")
    
    # Copy tokenizer if needed
    tokenizer_src = Path("Qwen3-TTS-Tokenizer-12Hz/model.safetensors")
    speech_tokenizer_dst = Path(model_path) / "speech_tokenizer" / "model.safetensors"
    if tokenizer_src.exists() and not speech_tokenizer_dst.exists():
        speech_tokenizer_dst.parent.mkdir(exist_ok=True)
        shutil.copy(tokenizer_src, speech_tokenizer_dst)
    
    preferred_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    last_err = None
    
    for tdtype in preferred_dtypes:
        try:
            m = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map="mps",
                torch_dtype=tdtype,
            )
            try:
                m.model.eval()
            except Exception:
                pass
            loaded_models[model_name] = m
            print(f"{model_name} loaded with {tdtype}!")
            return m
        except Exception as e:
            last_err = e
            _mps_cleanup()
    
    raise RuntimeError(f"Failed to load {model_name}: {last_err}")


if __name__ == "__main__":
    # Test model loading
    print("Testing model loader...")
    try:
        model = get_model("1.7B-CustomVoice")
        print(f"Model loaded successfully!")
        print(f"Supported speakers: {model.get_supported_speakers()}")
    except Exception as e:
        print(f"Error: {e}")
