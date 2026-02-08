#!/usr/bin/env python3
"""
Local RunPod handler test – simulates a RunPod job without the SDK.

Usage:
    python scripts/local_runpod_test.py                       # TTS test
    python scripts/local_runpod_test.py --text "Say this"     # custom text
    python scripts/local_runpod_test.py --segments            # multi-speaker TTS
    python scripts/local_runpod_test.py --podcast             # full podcast pipeline
    python scripts/local_runpod_test.py --payload job.json    # custom JSON
    python scripts/local_runpod_test.py --save out.wav        # save to file
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── TTS jobs ──────────────────────────────────────────────────────

def build_single_tts_job(text: str, voice: str, model: str, fmt: str) -> dict:
    return {
        "id": "local-tts-single",
        "input": {
            "action": "tts",
            "text": text,
            "voice": voice,
            "model": model,
            "output_format": fmt,
            "params": {"temperature": 0.3, "top_k": 50, "top_p": 0.85, "language": "en"},
        },
    }


def build_segments_tts_job(model: str, fmt: str) -> dict:
    return {
        "id": "local-tts-segments",
        "input": {
            "action": "tts",
            "segments": [
                {"text": "Welcome to our podcast! Today we're discussing the future of AI.", "voice": "serena"},
                {"text": "Thanks for having me. I'm excited to share my thoughts on this topic.", "voice": "ryan"},
                {"text": "Let's start with the big picture. Where do you see AI headed?", "voice": "serena"},
            ],
            "model": model,
            "output_format": fmt,
            "params": {"temperature": 0.3, "top_k": 50, "top_p": 0.85, "language": "en"},
        },
    }


# ── Podcast job ───────────────────────────────────────────────────

def build_podcast_job(fmt: str) -> dict:
    """Full pipeline: LLM → outline → transcript → TTS → combined audio."""
    return {
        "id": "local-podcast",
        "input": {
            "action": "podcast",
            "topic": "The future of artificial intelligence in creative industries",
            "key_points": [
                "How AI is changing music and art",
                "The ethics of AI-generated content",
                "Opportunities for creators",
            ],
            "briefing": "Keep it conversational, insightful, and accessible to a general audience.",
            "num_segments": 2,
            "language": "English",
            "quality_preset": "standard",
            "voices": [
                {"voice_id": "serena", "role": "Host", "type": "preset", "name": "Sarah"},
                {"voice_id": "ryan", "role": "Expert", "type": "preset", "name": "Ryan"},
            ],
            "llm": {
                "provider": os.environ.get("LLM_PROVIDER", "openrouter"),
                "model": os.environ.get("LLM_MODEL", "google/gemini-2.5-flash"),
                "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            },
            "output_format": fmt,
        },
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local RunPod handler test")
    parser.add_argument("--text", type=str, default="Hello! This is a test of the Qwen3 text to speech system.")
    parser.add_argument("--voice", type=str, default="serena")
    parser.add_argument("--model", type=str, default="1.7B-CustomVoice")
    parser.add_argument("--format", type=str, default="wav", choices=["wav", "mp3"])
    parser.add_argument("--segments", action="store_true", help="Multi-speaker TTS test")
    parser.add_argument("--podcast", action="store_true", help="Full podcast pipeline test")
    parser.add_argument("--save", type=str, default=None, help="Save audio to file")
    parser.add_argument("--payload", type=str, default=None, help="Custom JSON payload file")
    args = parser.parse_args()

    # Build job
    if args.payload:
        with open(args.payload) as f:
            job = json.load(f)
        if "id" not in job:
            job["id"] = "local-custom"
        if "input" not in job:
            job = {"id": "local-custom", "input": job}
    elif args.podcast:
        job = build_podcast_job(args.format)
    elif args.segments:
        job = build_segments_tts_job(args.model, args.format)
    else:
        job = build_single_tts_job(args.text, args.voice, args.model, args.format)

    action = job.get("input", {}).get("action", "tts")
    print("=" * 60)
    print(f"LOCAL RUNPOD TEST — action: {action}")
    print("=" * 60)
    print(f"\nJob payload:\n{json.dumps(job, indent=2)}\n")

    from handler import handler

    print("Running handler …\n")
    t0 = time.time()
    result = handler(job)
    elapsed = time.time() - t0

    if "error" in result:
        print(f"\n❌ ERROR: {result['error']}")
        if "traceback" in result:
            print(f"\nTraceback:\n{result['traceback']}")
        sys.exit(1)

    meta = result.get("metadata", {})
    print(f"\n✅ Success!")
    print(f"   Duration:        {meta.get('duration_seconds', '?')}s")
    print(f"   Sample rate:     {meta.get('sample_rate', '?')} Hz")
    print(f"   Format:          {meta.get('format', '?')}")
    print(f"   Generation time: {meta.get('generation_time_seconds', round(elapsed, 3))}s")

    if meta.get("num_dialogue_lines"):
        print(f"   Dialogue lines:  {meta['num_dialogue_lines']}")
    if meta.get("num_outline_segments"):
        print(f"   Outline segs:    {meta['num_outline_segments']}")
    if meta.get("num_segments"):
        print(f"   TTS segments:    {meta['num_segments']}")

    if "audio_url" in result:
        print(f"   Audio URL:       {result['audio_url']}")

    # Show transcript/outline if present
    if "transcript" in result:
        dialogues = result["transcript"].get("dialogues", [])
        print(f"\n   Transcript ({len(dialogues)} lines):")
        for d in dialogues[:5]:
            text_preview = d["text"][:60] + "…" if len(d["text"]) > 60 else d["text"]
            print(f"     {d['speaker']}: {text_preview}")
        if len(dialogues) > 5:
            print(f"     ... and {len(dialogues) - 5} more lines")

    # Save audio
    if "audio_base64" in result:
        audio_bytes = base64.b64decode(result["audio_base64"])
        size_kb = len(audio_bytes) / 1024

        save_path = args.save or f"test_output.{meta.get('format', 'wav')}"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(audio_bytes)

        print(f"   File size:       {size_kb:.1f} KB")
        print(f"   Saved to:        {os.path.abspath(save_path)}")

    print(f"\nTotal wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
