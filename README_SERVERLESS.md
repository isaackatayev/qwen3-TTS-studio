# Qwen3-TTS — RunPod Serverless Deployment

Scale-to-zero, pay-per-job TTS via [RunPod Serverless](https://docs.runpod.io/serverless/overview). Models load once per warm worker and stay cached. No Gradio UI runs in serverless mode.

---

## Quick Start

### 1. Build the Docker image

```bash
docker build -f Dockerfile.serverless -t qwen3-tts-serverless .
```

### 2. Provide models

Models are large (~3-7 GB each). Mount them or bake them into the image:

| Method | How |
|--------|-----|
| **RunPod Network Volume** (recommended) | Create a volume, download models there, attach to your endpoint. Set `QWEN_TTS_MODEL_DIR=/runpod-volume/models`. |
| **Bake into image** | `COPY` the model directories into `/models/` in your Dockerfile. Increases image size but simplifies deployment. |
| **Download at startup** | Add a startup script that pulls from HuggingFace. Slow cold starts. |

Expected directory layout inside `QWEN_TTS_MODEL_DIR`:

```
/models/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
├── Qwen3-TTS-12Hz-0.6B-CustomVoice/
├── Qwen3-TTS-12Hz-1.7B-Base/
├── Qwen3-TTS-12Hz-0.6B-Base/
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
└── Qwen3-TTS-Tokenizer-12Hz/
    └── model.safetensors
```

### 3. Deploy on RunPod

1. Push image to Docker Hub / GHCR.
2. Create a **Serverless Endpoint** on RunPod.
3. Set the Docker image and environment variables (see below).
4. Configure GPU type (A100 / A40 / L40S recommended).

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QWEN_TTS_MODEL_DIR` | Yes | `/models` | Base directory containing model folders |
| `QWEN_TTS_DEVICE` | No | auto-detect | Force device (`cuda:0`, `cpu`) |
| `QWEN_TTS_MIN_NEW_TOKENS` | No | `60` | Min tokens to prevent audio truncation |
| `S3_BUCKET` | No | — | S3 bucket for audio upload |
| `S3_ACCESS_KEY` | No | — | S3 access key |
| `S3_SECRET_KEY` | No | — | S3 secret key |
| `S3_ENDPOINT` | No | — | S3 endpoint (for R2, MinIO, etc.) |
| `S3_REGION` | No | `us-east-1` | S3 region |

> When `S3_BUCKET` + credentials are set, audio is uploaded and a presigned URL is returned. Otherwise, audio is returned as base64.

---

## API Reference

### Input Schema

POST to your RunPod endpoint's `/run` or `/runsync` URL.

#### Single text

```json
{
  "input": {
    "text": "Hello world! This is Qwen3-TTS running on RunPod.",
    "voice": "male_1",
    "model": "1.7B-CustomVoice",
    "output_format": "wav",
    "params": {
      "temperature": 0.3,
      "top_k": 50,
      "top_p": 0.85,
      "language": "en"
    }
  }
}
```

#### Multi-speaker segments (podcast)

```json
{
  "input": {
    "segments": [
      {
        "text": "Welcome to our podcast! Today we discuss the future of AI.",
        "voice": "male_1"
      },
      {
        "text": "Thanks for having me. I think we're at an inflection point.",
        "voice": "female_1"
      },
      {
        "text": "That's a great perspective. Let's dig deeper.",
        "voice": "male_1"
      }
    ],
    "model": "1.7B-CustomVoice",
    "output_format": "mp3",
    "params": {
      "temperature": 0.3,
      "top_k": 50,
      "top_p": 0.85,
      "language": "en"
    }
  }
}
```

#### Input fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes* | — | Text to synthesize (use this OR `segments`) |
| `segments` | array | Yes* | — | Array of `{text, voice}` objects |
| `voice` | string | No | `male_1` | Default preset voice name |
| `model` | string | No | `1.7B-CustomVoice` | Model variant to use |
| `output_format` | string | No | `wav` | `wav` or `mp3` |
| `params` | object | No | see below | Generation parameters |

\* One of `text` or `segments` is required.

#### Params object

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float | 0.3 | Sampling temperature |
| `top_k` | int | 50 | Top-K sampling |
| `top_p` | float | 0.85 | Nucleus sampling |
| `repetition_penalty` | float | 1.0 | Repetition penalty |
| `max_new_tokens` | int | 1024 | Max tokens per chunk |
| `subtalker_temperature` | float | 0.3 | Sub-talker temperature |
| `subtalker_top_k` | int | 50 | Sub-talker Top-K |
| `subtalker_top_p` | float | 0.85 | Sub-talker Top-P |
| `language` | string | `en` | Language code |
| `instruct` | string | null | Optional instruction |

### Output Schema

#### Success (base64)

```json
{
  "audio_base64": "<base64-encoded audio bytes>",
  "metadata": {
    "duration_seconds": 3.456,
    "sample_rate": 24000,
    "format": "wav",
    "num_segments": 1,
    "generation_time_seconds": 2.1
  }
}
```

#### Success (S3 upload)

```json
{
  "audio_url": "https://your-bucket.s3.amazonaws.com/tts-output/abc123.wav?...",
  "metadata": {
    "duration_seconds": 3.456,
    "sample_rate": 24000,
    "format": "wav",
    "num_segments": 1,
    "generation_time_seconds": 2.1
  }
}
```

#### Error

```json
{
  "error": "Generation failed on segment 2: ...",
  "traceback": "..."
}
```

---

## Chunking / Long-form Audio

Long text is automatically split into ~120-character chunks (sentence-boundary aware) by the existing `audio.generator` module. Each chunk is generated independently with retry logic (up to 3 retries per chunk). Chunks are crossfaded and concatenated into one seamless output.

For 30-minute podcasts, use the `segments` input to pass each dialogue line as a separate segment with its own voice. The handler generates each segment sequentially, concatenates, and returns one file.

---

## Example: Call from Python

```python
import requests, json, base64

RUNPOD_API_KEY = "your-api-key"
ENDPOINT_ID = "your-endpoint-id"

resp = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "input": {
            "text": "Hello from Qwen3-TTS on RunPod!",
            "voice": "male_1",
            "model": "1.7B-CustomVoice",
            "output_format": "wav",
        }
    },
    timeout=120,
)

data = resp.json()["output"]
audio_bytes = base64.b64decode(data["audio_base64"])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
print(f"Saved {data['metadata']['duration_seconds']}s of audio")
```

---

## Example: cURL

```bash
curl -s -X POST \
  "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello from Qwen3-TTS!",
      "voice": "male_1",
      "model": "1.7B-CustomVoice",
      "output_format": "wav"
    }
  }' | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)['output']
audio = base64.b64decode(data['audio_base64'])
open('output.wav', 'wb').write(audio)
print(f'Saved {len(audio)} bytes')
"
```

---

## Local Testing

### Without RunPod SDK

```bash
# Set model directory (if models aren't in current dir)
export QWEN_TTS_MODEL_DIR=/path/to/models

# Run the local test script
python scripts/local_runpod_test.py

# Custom text
python scripts/local_runpod_test.py --text "Custom text here" --voice male_1

# Multi-speaker segments
python scripts/local_runpod_test.py --segments

# Save to specific file
python scripts/local_runpod_test.py --save my_audio.wav

# Use a custom JSON payload
python scripts/local_runpod_test.py --payload my_job.json

# MP3 output
python scripts/local_runpod_test.py --format mp3
```

### With RunPod SDK (simulated)

```bash
pip install runpod
python handler.py
# In another terminal, send test requests to the local RunPod test server
```

---

## Available Preset Voices

The available voices depend on the model variant. For `1.7B-CustomVoice` / `0.6B-CustomVoice`, run:

```python
from audio.model_loader import get_model
model = get_model("1.7B-CustomVoice")
print(model.get_supported_speakers())
```

Common presets include: `male_1`, `male_2`, `female_1`, `female_2`, etc.

---

## Architecture

```
RunPod Job
    │
    ▼
handler.py  ─── handler(job)
    │
    ├── audio.model_loader.get_model()   ← cached per warm worker
    │
    ├── audio.generator._generate_preset_voice()
    │   └── _split_text_into_chunks()    ← auto-chunking
    │   └── model.generate_custom_voice()
    │   └── retry logic (3 attempts)
    │
    ├── crossfade & concatenate segments
    │
    ├── encode (WAV/MP3)
    │
    └── S3 upload (if configured) → audio_url
        OR base64 encode           → audio_base64
```

No Gradio, no FastAPI. Pure RunPod `handler(job)` function.
