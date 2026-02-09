"""
RunPod Serverless handler for Qwen3-TTS Studio.

Two modes:
  1. "tts"     – text-to-speech only (default)
  2. "podcast" – full pipeline: LLM outline → transcript → TTS → combined audio

Models are loaded once per warm worker (global cache).
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `audio.*` / `podcast.*` imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Lazy-import flag
# ---------------------------------------------------------------------------
_imports_ready = False


def _ensure_imports():
    global _imports_ready
    if _imports_ready:
        return
    import audio.model_loader  # noqa: F401
    import audio.generator  # noqa: F401

    _imports_ready = True


# ---------------------------------------------------------------------------
# S3 upload helper
# ---------------------------------------------------------------------------

def _s3_upload(data: bytes, key: str, content_type: str) -> str | None:
    """Upload bytes to S3-compatible storage; return presigned URL or None."""
    bucket = os.environ.get("S3_BUCKET", "").strip()
    access_key = os.environ.get("S3_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("S3_SECRET_KEY", "").strip()

    if not (bucket and access_key and secret_key):
        return None

    try:
        import boto3
        from botocore.config import Config as BotoConfig

        endpoint = os.environ.get("S3_ENDPOINT", "").strip() or None
        region = os.environ.get("S3_REGION", "").strip() or "us-east-1"

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=BotoConfig(signature_version="s3v4"),
        )
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
        return url
    except Exception as exc:
        print(f"[S3] Upload failed: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Default TTS params & helpers
# ---------------------------------------------------------------------------

MAX_SEGMENT_RETRIES = 3
RETRY_DELAYS = (2, 5, 10)

DEFAULT_TTS_PARAMS: dict[str, Any] = {
    "temperature": 0.3,
    "top_k": 50,
    "top_p": 0.85,
    "repetition_penalty": 1.0,
    "max_new_tokens": 1024,
    "subtalker_temperature": 0.3,
    "subtalker_top_k": 50,
    "subtalker_top_p": 0.85,
    "language": "en",
    "instruct": None,
}


def _merge_params(user_params: dict | None) -> dict[str, Any]:
    merged = dict(DEFAULT_TTS_PARAMS)
    if user_params:
        for k, v in user_params.items():
            if v is not None:
                merged[k] = v
    return merged


def _generate_single_segment(
    text: str, voice: str, model_name: str, params: dict[str, Any],
) -> tuple[np.ndarray, int]:
    from audio.model_loader import get_model
    from audio.generator import _generate_preset_voice

    model = get_model(model_name)
    last_err: Exception | None = None
    for attempt in range(MAX_SEGMENT_RETRIES + 1):
        try:
            wavs, sr = _generate_preset_voice(model, text, voice, params)
            return wavs[0], int(sr)
        except Exception as exc:
            last_err = exc
            if attempt < MAX_SEGMENT_RETRIES:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"[HANDLER] Segment retry {attempt + 1}: {exc}", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Segment failed after {MAX_SEGMENT_RETRIES + 1} attempts: {last_err}")


def _crossfade(a: np.ndarray, b: np.ndarray, sr: int, ms: int = 30) -> np.ndarray:
    from audio.generator import _crossfade_audio
    return _crossfade_audio(a, b, sr, fade_ms=ms)


def _encode_output(
    audio_bytes: bytes,
    ext: str,
    content_type: str,
    duration: float,
    sr: int,
    job_id: str,
    extra_meta: dict | None = None,
) -> dict[str, Any]:
    """Build response dict with audio_url or audio_base64."""
    metadata: dict[str, Any] = {
        "duration_seconds": round(duration, 3),
        "sample_rate": sr,
        "format": ext,
    }
    if extra_meta:
        metadata.update(extra_meta)

    s3_key = f"tts-output/{job_id}.{ext}"
    url = _s3_upload(audio_bytes, s3_key, content_type)
    if url:
        return {"audio_url": url, "metadata": metadata}
    else:
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return {"audio_base64": b64, "metadata": metadata}


def _audio_to_bytes(audio: np.ndarray, sr: int, output_format: str) -> tuple[bytes, str, str]:
    """Convert numpy audio to bytes in the requested format."""
    buf = io.BytesIO()
    if output_format == "mp3":
        try:
            from pydub import AudioSegment
            wav_buf = io.BytesIO()
            sf.write(wav_buf, audio, sr, format="WAV")
            wav_buf.seek(0)
            seg = AudioSegment.from_wav(wav_buf)
            seg.export(buf, format="mp3", bitrate="192k")
            return buf.getvalue(), "mp3", "audio/mpeg"
        except ImportError:
            print("[HANDLER] pydub not installed, falling back to WAV", flush=True)

    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue(), "wav", "audio/wav"


# ===================================================================
# ACTION: tts  (text-to-speech only)
# ===================================================================

def _handle_tts(job_id: str, inp: dict) -> dict[str, Any]:
    _ensure_imports()

    text: str | None = inp.get("text")
    segments: list[dict] | None = inp.get("segments")
    if not text and not segments:
        return {"error": "Either 'text' or 'segments' is required."}

    voice: str = inp.get("voice", "male_1")
    model_name: str = inp.get("model", "1.7B-CustomVoice")
    output_format: str = inp.get("output_format", "wav").lower()
    params = _merge_params(inp.get("params"))

    if output_format not in ("wav", "mp3"):
        return {"error": f"Unsupported output_format: {output_format}. Use 'wav' or 'mp3'."}

    seg_list = (
        [{"text": s.get("text", ""), "voice": s.get("voice", voice)} for s in segments if s.get("text", "").strip()]
        if segments
        else [{"text": text, "voice": voice}]
    )
    if not seg_list:
        return {"error": "No non-empty segments to generate."}

    print(f"[TTS] Job {job_id}: {len(seg_list)} segment(s), model={model_name}", flush=True)
    t0 = time.time()

    all_audio: list[np.ndarray] = []
    sr = 0
    for i, seg in enumerate(seg_list):
        print(f"[TTS] Segment {i+1}/{len(seg_list)}: voice={seg['voice']}, chars={len(seg['text'])}", flush=True)
        try:
            arr, seg_sr = _generate_single_segment(seg["text"], seg["voice"], model_name, params)
        except Exception as exc:
            return {"error": f"TTS failed on segment {i+1}: {exc}", "traceback": traceback.format_exc()}
        if sr == 0:
            sr = seg_sr
        all_audio.append(arr)

    merged = all_audio[0]
    for a in all_audio[1:]:
        merged = _crossfade(merged, a, sr)

    elapsed = time.time() - t0
    audio_bytes, ext, ct = _audio_to_bytes(merged, sr, output_format)
    return _encode_output(audio_bytes, ext, ct, len(merged) / sr, sr, job_id, {
        "num_segments": len(seg_list),
        "generation_time_seconds": round(elapsed, 3),
    })


# ===================================================================
# ACTION: podcast  (full LLM → outline → transcript → TTS → combine)
# ===================================================================

def _handle_podcast(job_id: str, inp: dict) -> dict[str, Any]:
    """Run the full podcast generation pipeline."""
    _ensure_imports()

    # ---- Required fields ----
    topic = str(inp.get("topic", "")).strip()
    if not topic:
        return {"error": "'topic' is required for podcast generation."}

    voices_raw = inp.get("voices")
    if not voices_raw or not isinstance(voices_raw, list):
        return {"error": "'voices' array is required (list of {voice_id, role, type, name})."}

    # ---- Optional fields ----
    key_points = inp.get("key_points", [])
    if isinstance(key_points, str):
        key_points = [k.strip() for k in key_points.split("\n") if k.strip()]
    briefing = str(inp.get("briefing", "")).strip()
    num_segments = int(inp.get("num_segments", 3))
    language = str(inp.get("language", "English")).strip()
    quality_preset = inp.get("quality_preset", "standard")
    output_format = str(inp.get("output_format", "mp3")).lower()

    # ---- LLM config ----
    llm_raw = inp.get("llm", {})
    if not isinstance(llm_raw, dict):
        return {"error": "'llm' must be an object with provider, model, api_key."}

    llm_provider_str = str(llm_raw.get("provider", "openrouter")).lower()
    llm_model = str(llm_raw.get("model", "")).strip()
    llm_api_key = str(llm_raw.get("api_key", "")).strip()

    # Allow API key from env vars as fallback
    if not llm_api_key:
        env_key_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "ollama": "",
        }
        env_var = env_key_map.get(llm_provider_str, "")
        if env_var:
            llm_api_key = os.environ.get(env_var, "").strip()
        if not llm_api_key and llm_provider_str != "ollama":
            return {"error": f"LLM API key required. Pass in llm.api_key or set {env_var} env var."}

    from podcast.llm_client import LLMProvider, get_default_config

    provider_map = {
        "openai": LLMProvider.OPENAI,
        "openrouter": LLMProvider.OPENROUTER,
        "claude": LLMProvider.CLAUDE,
        "ollama": LLMProvider.OLLAMA,
    }
    provider = provider_map.get(llm_provider_str)
    if provider is None:
        return {"error": f"Unknown LLM provider: {llm_provider_str}. Use openai/openrouter/claude/ollama."}

    llm_config = get_default_config(
        provider=provider,
        api_key=llm_api_key,
        model=llm_model,
    )

    # ---- Run the orchestrator ----
    from podcast.orchestrator import generate_podcast

    print(f"[PODCAST] Job {job_id}: topic={topic!r}, segments={num_segments}, provider={llm_provider_str}", flush=True)
    t0 = time.time()

    content_input: dict[str, object] = {
        "topic": topic,
        "key_points": key_points,
        "briefing": briefing,
        "num_segments": num_segments,
        "language": language,
    }

    try:
        artifacts = generate_podcast(
            content_input=content_input,
            voice_selections=voices_raw,
            quality_preset=quality_preset,
            llm_config=llm_config,
        )
    except Exception as exc:
        return {"error": f"Podcast generation failed: {exc}", "traceback": traceback.format_exc()}

    elapsed = time.time() - t0
    print(f"[PODCAST] Job {job_id} pipeline done in {elapsed:.1f}s", flush=True)

    # ---- Read the combined audio file and return ----
    combined_path = artifacts.get("combined_audio_path", "")
    if not combined_path or not Path(combined_path).exists():
        return {"error": "Pipeline completed but combined audio file not found."}

    combined_audio_data, sr = sf.read(combined_path)
    audio_bytes, ext, ct = _audio_to_bytes(combined_audio_data, sr, output_format)
    duration = len(combined_audio_data) / sr

    # Read transcript for metadata
    transcript_data = None
    transcript_path = artifacts.get("transcript_path", "")
    if transcript_path and Path(transcript_path).exists():
        import json
        transcript_data = json.loads(Path(transcript_path).read_text())

    outline_data = None
    outline_path = artifacts.get("outline_path", "")
    if outline_path and Path(outline_path).exists():
        import json
        outline_data = json.loads(Path(outline_path).read_text())

    result = _encode_output(audio_bytes, ext, ct, duration, sr, job_id, {
        "generation_time_seconds": round(elapsed, 3),
        "num_dialogue_lines": len(transcript_data.get("dialogues", [])) if transcript_data else 0,
        "num_outline_segments": len(outline_data.get("segments", [])) if outline_data else 0,
    })

    # Include transcript & outline in response for inspection
    if transcript_data:
        result["transcript"] = transcript_data
    if outline_data:
        result["outline"] = outline_data

    # Cleanup temp podcast dir
    podcast_dir = artifacts.get("podcast_dir", "")
    if podcast_dir and Path(podcast_dir).exists():
        import shutil
        shutil.rmtree(podcast_dir, ignore_errors=True)

    return result


# ===================================================================
# ACTION: debug (filesystem inspection)
# ===================================================================

def _handle_debug() -> dict[str, Any]:
    """List filesystem paths so we can find where the volume is mounted."""
    import subprocess
    info: dict[str, Any] = {
        "QWEN_TTS_MODEL_DIR": os.environ.get("QWEN_TTS_MODEL_DIR", "(not set)"),
        "cwd": os.getcwd(),
    }

    # Check common mount points
    for path in ["/runpod-volume", "/workspace", "/models", "/mnt", "/data"]:
        try:
            if os.path.exists(path):
                contents = os.listdir(path)
                info[path] = contents[:20]
                # Go one level deeper for model dirs
                for item in contents:
                    sub = os.path.join(path, item)
                    if os.path.isdir(sub):
                        try:
                            info[f"{path}/{item}"] = os.listdir(sub)[:10]
                        except Exception:
                            pass
            else:
                info[path] = "(does not exist)"
        except Exception as e:
            info[path] = f"(error: {e})"

    # Also check root-level dirs
    try:
        info["/"] = sorted(os.listdir("/"))
    except Exception:
        pass

    return info


# ===================================================================
# Main handler
# ===================================================================

def handler(job: dict) -> dict[str, Any]:
    """RunPod Serverless handler.

    Dispatches to _handle_tts or _handle_podcast based on ``input.action``.

    Actions:
      - ``"tts"`` (default): text-to-speech only
      - ``"podcast"``: full pipeline (LLM → outline → transcript → TTS → combine)
    """
    job_id = job.get("id", "local")
    inp = job.get("input", {})
    action = str(inp.get("action", "tts")).lower()

    if action == "debug":
        return _handle_debug()
    elif action == "podcast":
        return _handle_podcast(job_id, inp)
    elif action == "tts":
        return _handle_tts(job_id, inp)
    else:
        return {"error": f"Unknown action: {action}. Use 'tts', 'podcast', or 'debug'."}


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    try:
        import runpod
        print("[HANDLER] Starting RunPod serverless worker …", flush=True)
        runpod.serverless.start({"handler": handler})
    except ImportError:
        print(
            "[HANDLER] runpod not installed. For local testing:\n"
            "  python scripts/local_runpod_test.py\n"
            "  python scripts/local_runpod_test.py --podcast",
            flush=True,
        )
        sys.exit(1)
