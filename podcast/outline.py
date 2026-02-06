"""LLM-powered outline generator for podcast episodes."""

# pyright: reportImplicitRelativeImport=false, reportMissingImports=false, reportDeprecated=false
# pyright: reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportAny=false, reportImplicitStringConcatenation=false
# pyright: reportAssignmentType=false

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from importlib import import_module
from typing import Protocol, cast

from pydantic import ValidationError

from podcast.llm_client import LLMConfig, LLMProvider, chat_completion, create_llm_client, get_default_config
from podcast.prompts import format_persona_context
from storage.persona_models import Persona

_models_module = import_module("podcast.models")


class _Speaker(Protocol):
    name: str
    voice_id: str
    role: str
    type: str


class _SpeakerProfile(Protocol):
    speakers: Sequence[_Speaker]


class _Segment(Protocol):
    size: str


class _Outline(Protocol):
    segments: Sequence[_Segment]

    @classmethod
    def model_validate(cls, data: object) -> "_Outline":
        ...


OutlineClass = cast(type[_Outline], getattr(_models_module, "Outline"))
SpeakerProfileClass = cast(
    type[_SpeakerProfile], getattr(_models_module, "SpeakerProfile")
)


def _coerce_speakers(
    speakers: Sequence[_Speaker] | _SpeakerProfile,
) -> list[_Speaker]:
    if isinstance(speakers, SpeakerProfileClass):
        return list(speakers.speakers)
    return list(cast(Sequence[_Speaker], speakers))


def _format_key_points(key_points: Iterable[str]) -> str:
    cleaned = [point.strip() for point in key_points if point.strip()]
    if not cleaned:
        return "- None provided"
    return "\n".join(f"- {point}" for point in cleaned)


def _format_speakers(speakers: Sequence[_Speaker]) -> str:
    if not speakers:
        return "- None provided"
    lines: list[str] = []
    for speaker in speakers:
        lines.append(
            "- {name} (role: {role}, voice_id: {voice_id}, type: {type})".format(
                name=speaker.name,
                role=speaker.role,
                voice_id=speaker.voice_id,
                type=speaker.type,
            )
        )
    return "\n".join(lines)


_format_persona_context = format_persona_context


def _segment_size_targets(num_segments: int) -> dict[str, int]:
    targets = {"short": 0.3, "medium": 0.5, "long": 0.2}
    raw = {key: num_segments * ratio for key, ratio in targets.items()}
    counts = {key: int(raw[key]) for key in targets}
    remainder = num_segments - sum(counts.values())
    priority = {"medium": 2, "short": 1, "long": 0}
    ordering = sorted(
        targets.keys(),
        key=lambda key: (raw[key] - counts[key], priority[key]),
        reverse=True,
    )
    for index in range(remainder):
        counts[ordering[index % len(ordering)]] += 1
    return counts


def _format_size_targets(size_targets: dict[str, int]) -> str:
    return "\n".join(
        f"- {size}: {count}" for size, count in size_targets.items()
    )


def _build_outline_prompt(
    topic: str,
    key_points: list[str],
    briefing: str,
    num_segments: int,
    speakers: Sequence[_Speaker],
    size_targets: dict[str, int],
    personas: dict[str, Persona] | None = None,
) -> str:
    outline_schema = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "size": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                        },
                    },
                    "required": ["title", "description", "size"],
                },
                "minItems": num_segments,
                "maxItems": num_segments,
            }
        },
        "required": ["segments"],
    }

    persona_section = ""
    if personas:
        persona_context = _format_persona_context(personas)
        if persona_context:
            persona_section = f"""
SPEAKER PERSONAS:
{persona_context}
"""

    return f"""You are an expert podcast producer. Build a clear, engaging episode outline.

TOPIC:
{topic}

KEY POINTS:
{_format_key_points(key_points)}

STYLE BRIEFING:
{briefing}

SPEAKERS:
{_format_speakers(speakers)}
{persona_section}
REQUIREMENTS:
1. Create exactly {num_segments} segments.
2. Ensure every key point is covered across the outline.
3. Segment sizes must follow this distribution (use each size exactly this many times):
{_format_size_targets(size_targets)}
4. Segment order should flow logically and build momentum.
5. Include speaker roles when describing the segment focus.
6. Design segments that leverage each speaker's expertise and personality.

OUTPUT FORMAT:
Return ONLY valid JSON matching this schema:
{json.dumps(outline_schema, indent=2)}
"""


def _validate_distribution(outline: _Outline, size_targets: dict[str, int]) -> None:
    counts = {"short": 0, "medium": 0, "long": 0}
    for segment in outline.segments:
        if segment.size not in counts:
            raise ValueError(f"Unexpected segment size: {segment.size}")
        counts[segment.size] += 1
    if counts != size_targets:
        raise ValueError(
            f"Segment sizes do not match distribution. Expected {size_targets}, got {counts}."
        )


def _parse_outline_response(
    response_text: str,
    num_segments: int,
    size_targets: dict[str, int],
) -> _Outline:
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Invalid JSON response from LLM.") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    payload = cast(dict[str, object], payload)

    try:
        outline = OutlineClass.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("LLM response does not match Outline schema.") from exc

    if len(outline.segments) != num_segments:
        raise ValueError(
            f"Segment count mismatch. Expected {num_segments}, got {len(outline.segments)}."
        )

    _validate_distribution(outline, size_targets)
    return outline


def generate_outline(
    topic: str,
    key_points: list[str],
    briefing: str,
    num_segments: int,
    speakers: Sequence[_Speaker] | _SpeakerProfile,
    personas: dict[str, Persona] | None = None,
    llm_config: LLMConfig | None = None,
) -> _Outline:
    """Generate a podcast outline using the configured LLM provider."""
    if num_segments < 1:
        raise ValueError("num_segments must be at least 1.")

    speaker_list = _coerce_speakers(speakers)
    size_targets = _segment_size_targets(num_segments)
    prompt = _build_outline_prompt(
        topic=topic,
        key_points=key_points,
        briefing=briefing,
        num_segments=num_segments,
        speakers=speaker_list,
        size_targets=size_targets,
        personas=personas,
    )

    if llm_config is None:
        # Fallback to OpenAI for backward compatibility
        from config import get_openai_api_key
        llm_config = get_default_config(
            LLMProvider.OPENAI,
            api_key=get_openai_api_key(),
        )

    client = create_llm_client(llm_config)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            content = chat_completion(
                client=client,
                model=llm_config.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You create podcast outlines that are structured, "
                            "engaging, and JSON-only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=llm_config.temperature,
                json_mode=True,
                provider=llm_config.provider,
            )
            return _parse_outline_response(content, num_segments, size_targets)
        except ValueError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep((1, 2, 4)[attempt])

    raise RuntimeError(
        "Failed to generate outline after 3 attempts."
    ) from last_error
