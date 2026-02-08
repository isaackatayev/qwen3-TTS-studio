#!/usr/bin/env python3
"""
Local RunPod handler test – simulates a RunPod job without the RunPod SDK.

Usage:
    python scripts/local_runpod_test.py                    # default test
    python scripts/local_runpod_test.py --text "Say this"  # custom text
    python scripts/local_runpod_test.py --segments         # multi-speaker test
    python scripts/local_runpod_test.py --save out.wav     # save to file

Requires models to be available locally (set QWEN_TTS_MODEL_DIR if needed).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def build_single_job(text: str, voice: str, model: str, fmt: str) -> dict:
    """Build a single-text job payload."""
    return {
        "id": "local-test-single",
        "input": {
            "text": text,
            "voice": voice,
            "model": model,
            "output_format": fmt,
            "params": {
                "temperature": 0.3,
                "top_k": 50,
                "top_p": 0.85,
                "language": "en",
            },
        },
    }


def build_segments_job(model: str, fmt: str) -> dict:
    """Build a multi-segment (multi-speaker) job payload."""
    return {
        "id": "local-test-segments",
        "input": {
            "segments": [
                {
                    "text": "Welcome to our podcast! Today we're discussing the future of AI.",
                    "voice": "male_1",
                },
                {
                    "text": "Thanks for having me. I'm excited to share my thoughts on this topic.",
                    "voice": "female_1",
                },
                {
                    "text": "Let's start with the big picture. Where do you see AI headed in the next five years?",
                    "voice": "male_1",
                },
            ],
            "model": model,
            "output_format": fmt,
            "params": {
                "temperature": 0.3,
                "top_k": 50,
                "top_p": 0.85,
                "language": "en",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Local RunPod handler test")
    parser.add_argument("--text", type=str, default="Hello! This is a test of the Qwen3 text to speech system running on RunPod serverless.")
    parser.add_argument("--voice", type=str, default="male_1")
    parser.add_argument("--model", type=str, default="1.7B-CustomVoice")
    parser.add_argument("--format", type=str, default="wav", choices=["wav", "mp3"])
    parser.add_argument("--segments", action="store_true", help="Run multi-segment test")
    parser.add_argument("--save", type=str, default=None, help="Save audio to file path")
    parser.add_argument("--payload", type=str, default=None, help="Path to custom JSON payload file")
    args = parser.parse_args()

    # Build job
    if args.payload:
        with open(args.payload) as f:
            job = json.load(f)
        if "id" not in job:
            job["id"] = "local-test-custom"
        if "input" not in job:
            job = {"id": "local-test-custom", "input": job}
    elif args.segments:
        job = build_segments_job(args.model, args.format)
    else:
        job = build_single_job(args.text, args.voice, args.model, args.format)

    print("=" * 60)
    print("LOCAL RUNPOD HANDLER TEST")
    print("=" * 60)
    print(f"\nJob payload:\n{json.dumps(job, indent=2)}\n")

    # Import and run handler
    from handler import handler

    print("Running handler …\n")
    t0 = time.time()
    result = handler(job)
    elapsed = time.time() - t0

    # Check for errors
    if "error" in result:
        print(f"\n❌ ERROR: {result['error']}")
        if "traceback" in result:
            print(f"\nTraceback:\n{result['traceback']}")
        sys.exit(1)

    # Print metadata
    meta = result.get("metadata", {})
    print(f"\n✅ Success!")
    print(f"   Duration:        {meta.get('duration_seconds', '?')}s")
    print(f"   Sample rate:     {meta.get('sample_rate', '?')} Hz")
    print(f"   Format:          {meta.get('format', '?')}")
    print(f"   Segments:        {meta.get('num_segments', '?')}")
    print(f"   Generation time: {meta.get('generation_time_seconds', round(elapsed, 3))}s")

    if "audio_url" in result:
        print(f"   Audio URL:       {result['audio_url']}")

    # Save audio
    if "audio_base64" in result:
        audio_bytes = base64.b64decode(result["audio_base64"])
        size_kb = len(audio_bytes) / 1024

        save_path = args.save
        if not save_path:
            ext = meta.get("format", "wav")
            save_path = f"test_output.{ext}"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(audio_bytes)

        print(f"   File size:       {size_kb:.1f} KB")
        print(f"   Saved to:        {os.path.abspath(save_path)}")
    elif "audio_url" in result:
        print(f"   (Audio uploaded to S3, not saved locally)")

    print(f"\nTotal wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
