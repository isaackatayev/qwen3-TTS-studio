# Qwen3-TTS Studio — RunPod Serverless Deployment

Scale-to-zero, pay-per-job podcast generation via [RunPod Serverless](https://docs.runpod.io/serverless/overview).

Two modes:
- **`tts`** — text-to-speech only (fast, simple)
- **`podcast`** — full pipeline: LLM outline → transcript → TTS → combined audio

---

## Quick Start

### 1. Build & push the Docker image

```bash
docker build -f Dockerfile.serverless -t youruser/qwen3-tts-serverless .
docker push youruser/qwen3-tts-serverless:latest
```

### 2. Models on a Network Volume

Create a RunPod Network Volume, attach a temp pod, download:

```bash
cd /workspace
mkdir -p models && cd models
python3.13 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir='./Qwen3-TTS-12Hz-1.7B-CustomVoice')
snapshot_download('Qwen/Qwen3-TTS-Tokenizer-12Hz', local_dir='./Qwen3-TTS-Tokenizer-12Hz')
print('Done!')
"
```

Terminate the temp pod. Models persist on the volume.

### 3. Create Serverless Endpoint

| Setting | Value |
|---------|-------|
| Docker Image | `youruser/qwen3-tts-serverless:latest` |
| GPU | A40 / L40S / A100 |
| Network Volume | your volume |

**Environment Variables:**

| Variable | Required | Value |
|----------|----------|-------|
| `QWEN_TTS_MODEL_DIR` | **Yes** | `/workspace/models` |
| `OPENROUTER_API_KEY` | For podcast mode | your key |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QWEN_TTS_MODEL_DIR` | Yes | `/models` | Base dir containing model folders |
| `QWEN_TTS_DEVICE` | No | auto | Force device (`cuda:0`, `cpu`) |
| `OPENROUTER_API_KEY` | For podcast | — | OpenRouter API key (fallback) |
| `OPENAI_API_KEY` | For podcast | — | OpenAI API key (fallback) |
| `ANTHROPIC_API_KEY` | For podcast | — | Anthropic API key (fallback) |
| `S3_BUCKET` | No | — | S3 bucket for audio upload |
| `S3_ACCESS_KEY` | No | — | S3 access key |
| `S3_SECRET_KEY` | No | — | S3 secret key |
| `S3_ENDPOINT` | No | — | S3 endpoint (R2, MinIO) |
| `S3_REGION` | No | `us-east-1` | S3 region |

---

## API: Podcast Mode (Full Pipeline)

### Input

```json
{
  "input": {
    "action": "podcast",
    "topic": "The future of AI in creative industries",
    "key_points": [
      "How AI is changing music and art",
      "The ethics of AI-generated content",
      "Opportunities for creators"
    ],
    "briefing": "Conversational, insightful, accessible to general audience.",
    "num_segments": 3,
    "language": "English",
    "quality_preset": "standard",
    "voices": [
      {"voice_id": "serena", "role": "Host", "type": "preset", "name": "Sarah"},
      {"voice_id": "ryan", "role": "Expert", "type": "preset", "name": "Ryan"}
    ],
    "llm": {
      "provider": "openrouter",
      "model": "google/gemini-2.5-flash",
      "api_key": "sk-or-v1-..."
    },
    "output_format": "mp3"
  }
}
```

### Podcast Input Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `action` | string | Yes | — | Must be `"podcast"` |
| `topic` | string | Yes | — | Podcast topic |
| `key_points` | array/string | No | `[]` | Key points to cover |
| `briefing` | string | No | `""` | Style instructions for the LLM |
| `num_segments` | int | No | `3` | Number of outline segments |
| `language` | string | No | `"English"` | Language for transcript |
| `quality_preset` | string/object | No | `"standard"` | `"quick"`, `"standard"`, or `"premium"` |
| `voices` | array | Yes | — | Voice selections (see below) |
| `llm` | object | Yes | — | LLM config (see below) |
| `output_format` | string | No | `"mp3"` | `"wav"` or `"mp3"` |

### Voice Selection Object

```json
{"voice_id": "serena", "role": "Host", "type": "preset", "name": "Sarah"}
```

Available preset voices: `serena`, `ryan`, `vivian`, `aiden`, `dylan`, `eric`, `sohee`, `uncle_fu`, `ono_anna`

Roles: `Host`, `Expert`, `Guest`, `Narrator`

### LLM Config Object

```json
{"provider": "openrouter", "model": "google/gemini-2.5-flash", "api_key": "sk-or-..."}
```

Providers: `openrouter`, `openai`, `claude`, `ollama`

If `api_key` is omitted, falls back to the corresponding env var (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

### Podcast Output

```json
{
  "audio_base64": "<base64 MP3>",
  "metadata": {
    "duration_seconds": 145.2,
    "sample_rate": 24000,
    "format": "mp3",
    "generation_time_seconds": 82.3,
    "num_dialogue_lines": 18,
    "num_outline_segments": 3
  },
  "transcript": {
    "dialogues": [
      {"speaker": "Sarah", "text": "Welcome to our show..."},
      {"speaker": "Ryan", "text": "Thanks for having me..."}
    ]
  },
  "outline": {
    "segments": [
      {"title": "Introduction", "description": "...", "size": "short"}
    ]
  }
}
```

---

## API: TTS Mode (Text-to-Speech Only)

### Single Text

```json
{
  "input": {
    "action": "tts",
    "text": "Hello from Qwen3-TTS!",
    "voice": "serena",
    "model": "1.7B-CustomVoice",
    "output_format": "wav"
  }
}
```

### Multi-Speaker Segments

```json
{
  "input": {
    "action": "tts",
    "segments": [
      {"text": "Welcome to our show!", "voice": "serena"},
      {"text": "Thanks for having me.", "voice": "ryan"}
    ],
    "model": "1.7B-CustomVoice",
    "output_format": "mp3"
  }
}
```

### TTS Input Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `action` | string | No | `"tts"` | `"tts"` (default if omitted) |
| `text` | string | Yes* | — | Text to synthesize |
| `segments` | array | Yes* | — | Array of `{text, voice}` |
| `voice` | string | No | `"male_1"` | Default preset voice |
| `model` | string | No | `"1.7B-CustomVoice"` | Model variant |
| `output_format` | string | No | `"wav"` | `"wav"` or `"mp3"` |
| `params` | object | No | — | temperature, top_k, top_p, etc. |

### TTS Output

```json
{
  "audio_base64": "<base64>",
  "metadata": {
    "duration_seconds": 3.4,
    "sample_rate": 24000,
    "format": "wav",
    "num_segments": 1,
    "generation_time_seconds": 2.1
  }
}
```

---

## Example: Python Client

```python
import requests, json, base64

RUNPOD_API_KEY = "your-api-key"
ENDPOINT_ID = "your-endpoint-id"
URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

# Full podcast generation
resp = requests.post(URL,
    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"},
    json={
        "input": {
            "action": "podcast",
            "topic": "The future of AI",
            "num_segments": 2,
            "voices": [
                {"voice_id": "serena", "role": "Host", "type": "preset", "name": "Sarah"},
                {"voice_id": "ryan", "role": "Expert", "type": "preset", "name": "Ryan"},
            ],
            "llm": {
                "provider": "openrouter",
                "model": "google/gemini-2.5-flash",
                "api_key": "sk-or-v1-...",
            },
            "output_format": "mp3",
        }
    },
    timeout=600,
)

data = resp.json()["output"]
audio = base64.b64decode(data["audio_base64"])
with open("podcast.mp3", "wb") as f:
    f.write(audio)
print(f"Saved {data['metadata']['duration_seconds']}s podcast")
print(f"Transcript: {len(data['transcript']['dialogues'])} lines")
```

---

## Local Testing

```bash
# Set model directory
export QWEN_TTS_MODEL_DIR=/path/to/models

# TTS only
python scripts/local_runpod_test.py
python scripts/local_runpod_test.py --text "Custom text" --voice serena

# Multi-speaker TTS
python scripts/local_runpod_test.py --segments

# Full podcast pipeline (needs LLM API key)
export OPENROUTER_API_KEY=sk-or-v1-...
python scripts/local_runpod_test.py --podcast

# Custom payload
python scripts/local_runpod_test.py --payload my_job.json

# Save to specific file
python scripts/local_runpod_test.py --save output.mp3 --format mp3
```

---

## Architecture

```
RunPod Job (action: "podcast")
    │
    ├── podcast.orchestrator.generate_podcast()
    │   ├── LLM → generate outline (via OpenRouter/OpenAI/Claude)
    │   ├── LLM → generate transcript
    │   ├── TTS → generate audio clips (audio.batch)
    │   │   └── audio.model_loader.get_model()  ← cached per warm worker
    │   │   └── audio.generator per dialogue line
    │   └── audio.combiner → final MP3
    │
    ├── encode (WAV/MP3)
    └── S3 upload or base64

RunPod Job (action: "tts")
    │
    ├── audio.model_loader.get_model()  ← cached
    ├── audio.generator per segment
    ├── crossfade & concatenate
    ├── encode (WAV/MP3)
    └── S3 upload or base64
```
