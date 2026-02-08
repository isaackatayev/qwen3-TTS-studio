"""
RunPod Serverless handler for Qwen3-TTS.

Stateless JSON job handler that loads models once per warm worker (global
cache) and generates audio from text via the existing audio/ pipeline.

Accepts:
  - text (str) or segments (list[dict]) with per-segment voice/text
  - voice (preset name), model, params, output_format

Returns:
  - audio_base64 (or audio_url when S3 is configured)
  - metadata: duration_seconds, sample_rate, format, num_segments
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
# Ensure project root is on sys.path so `audio.*` imports resolve when
# handler.py is executed directly (e.g. `python handler.py`).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Audio generation imports (lazy-loaded on first job for faster cold start)
# ---------------------------------------------------------------------------
_generator_loaded = False


def _ensure_imports():
    """Lazy-import heavy modules so the handler file can be parsed quickly."""
    global _generator_loaded
    if _generator_loaded:
        return
    # These trigger model-framework imports (torch, transformers, etc.)
    import audio.model_loader  # noqa: F401
    import audio.generator  # noqa: F401

    _generator_loaded = True


# ---------------------------------------------------------------------------
# S3 upload helper (optional – returns None when not configured)
# ---------------------------------------------------------------------------

def _s3_upload(data: bytes, key: str, content_type: str) -> str | None:
    """Upload bytes to S3-compatible storage and return a public/signed URL.

    Required env vars:
      S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY
    Optional:
      S3_ENDPOINT (for R2/MinIO), S3_REGION (default us-east-1)

    Returns None if S3 is not configured.
    """
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

        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

        # Generate presigned URL (1 hour expiry)
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
# Core generation logic
# ---------------------------------------------------------------------------

# Retry config for individual segment generation
MAX_SEGMENT_RETRIES = 3
RETRY_DELAYS = (2, 5, 10)

# Default TTS params
DEFAULT_PARAMS: dict[str, Any] = {
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
    """Merge user-supplied params over defaults."""
    merged = dict(DEFAULT_PARAMS)
    if user_params:
        for k, v in user_params.items():
            if v is not None:
                merged[k] = v
    return merged


def _generate_single_segment(
    text: str,
    voice: str,
    model_name: str,
    params: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Generate audio for a single text segment with retry logic.

    Returns (audio_array, sample_rate).
    """
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
                print(
                    f"[HANDLER] Segment retry {attempt + 1}/{MAX_SEGMENT_RETRIES}: {exc}",
                    flush=True,
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Segment generation failed after {MAX_SEGMENT_RETRIES + 1} attempts: {last_err}"
    )


def _crossfade(a: np.ndarray, b: np.ndarray, sr: int, ms: int = 30) -> np.ndarray:
    """Crossfade two audio arrays."""
    from audio.generator import _crossfade_audio

    return _crossfade_audio(a, b, sr, fade_ms=ms)


def _encode_output(
    audio: np.ndarray,
    sr: int,
    output_format: str,
    job_id: str,
) -> dict[str, Any]:
    """Encode audio array to the requested format, upload to S3 or return base64.

    Returns dict with either audio_base64 or audio_url, plus metadata.
    """
    buf = io.BytesIO()

    if output_format == "mp3":
        # Write WAV to buffer first, then convert to MP3 via pydub/ffmpeg
        try:
            from pydub import AudioSegment

            wav_buf = io.BytesIO()
            sf.write(wav_buf, audio, sr, format="WAV")
            wav_buf.seek(0)
            seg = AudioSegment.from_wav(wav_buf)
            seg.export(buf, format="mp3", bitrate="192k")
            content_type = "audio/mpeg"
            ext = "mp3"
        except ImportError:
            # Fallback: return WAV if pydub not available
            print("[HANDLER] pydub not installed, falling back to WAV", flush=True)
            sf.write(buf, audio, sr, format="WAV")
            content_type = "audio/wav"
            ext = "wav"
    else:
        sf.write(buf, audio, sr, format="WAV")
        content_type = "audio/wav"
        ext = "wav"

    audio_bytes = buf.getvalue()
    duration = len(audio) / sr

    metadata = {
        "duration_seconds": round(duration, 3),
        "sample_rate": sr,
        "format": ext,
    }

    # Try S3 upload first
    s3_key = f"tts-output/{job_id}.{ext}"
    url = _s3_upload(audio_bytes, s3_key, content_type)

    if url:
        return {"audio_url": url, "metadata": metadata}
    else:
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return {"audio_base64": b64, "metadata": metadata}


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict[str, Any]:
    """RunPod Serverless handler entry point.

    Expected ``job["input"]`` schema:

    .. code-block:: json

        {
            "text": "Hello world",
            "voice": "male_1",
            "model": "1.7B-CustomVoice",
            "params": {
                "temperature": 0.3,
                "top_k": 50,
                "top_p": 0.85,
                "language": "en"
            },
            "output_format": "wav"
        }

    Or with multi-speaker segments:

    .. code-block:: json

        {
            "segments": [
                {"text": "Welcome!", "voice": "male_1"},
                {"text": "Thanks!", "voice": "female_1"}
            ],
            "model": "1.7B-CustomVoice",
            "params": {},
            "output_format": "mp3"
        }
    """
    _ensure_imports()

    job_id = job.get("id", "local")
    inp = job.get("input", {})

    # ---- Validate input ----
    text: str | None = inp.get("text")
    segments: list[dict] | None = inp.get("segments")

    if not text and not segments:
        return {"error": "Either 'text' or 'segments' is required."}

    voice: str = inp.get("voice", "male_1")
    model_name: str = inp.get("model", "1.7B-CustomVoice")
    output_format: str = inp.get("output_format", "wav").lower()
    user_params: dict | None = inp.get("params")
    params = _merge_params(user_params)

    if output_format not in ("wav", "mp3"):
        return {"error": f"Unsupported output_format: {output_format}. Use 'wav' or 'mp3'."}

    # ---- Build segment list ----
    if segments:
        seg_list = [
            {
                "text": s.get("text", ""),
                "voice": s.get("voice", voice),
            }
            for s in segments
            if s.get("text", "").strip()
        ]
    else:
        seg_list = [{"text": text, "voice": voice}]

    if not seg_list:
        return {"error": "No non-empty segments to generate."}

    # ---- Generate audio for each segment ----
    print(
        f"[HANDLER] Job {job_id}: {len(seg_list)} segment(s), model={model_name}, "
        f"format={output_format}",
        flush=True,
    )
    t0 = time.time()

    all_audio: list[np.ndarray] = []
    sr: int = 0

    for i, seg in enumerate(seg_list):
        seg_text = seg["text"]
        seg_voice = seg["voice"]
        print(
            f"[HANDLER] Segment {i + 1}/{len(seg_list)}: voice={seg_voice}, "
            f"chars={len(seg_text)}",
            flush=True,
        )
        try:
            audio_arr, seg_sr = _generate_single_segment(
                seg_text, seg_voice, model_name, params
            )
        except Exception as exc:
            return {
                "error": f"Generation failed on segment {i + 1}: {exc}",
                "traceback": traceback.format_exc(),
            }

        if sr == 0:
            sr = seg_sr
        all_audio.append(audio_arr)

    # ---- Concatenate segments ----
    if len(all_audio) == 1:
        merged = all_audio[0]
    else:
        merged = all_audio[0]
        for audio in all_audio[1:]:
            merged = _crossfade(merged, audio, sr)

    elapsed = time.time() - t0
    print(
        f"[HANDLER] Job {job_id} done: {len(merged)/sr:.1f}s audio in {elapsed:.1f}s",
        flush=True,
    )

    # ---- Encode & return ----
    result = _encode_output(merged, sr, output_format, job_id)
    result["metadata"]["num_segments"] = len(seg_list)
    result["metadata"]["generation_time_seconds"] = round(elapsed, 3)
    return result


# ---------------------------------------------------------------------------
# Entry point – RunPod serverless OR local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # When run directly, start the RunPod serverless worker.
    # For local testing without RunPod, use scripts/local_runpod_test.py instead.
    try:
        import runpod

        print("[HANDLER] Starting RunPod serverless worker …", flush=True)
        runpod.serverless.start({"handler": handler})
    except ImportError:
        print(
            "[HANDLER] runpod package not installed. "
            "Install with: pip install runpod\n"
            "For local testing, run: python scripts/local_runpod_test.py",
            flush=True,
        )
        sys.exit(1)
