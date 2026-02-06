"""LLM-powered transcript generator for podcast episodes."""

# pyright: reportImplicitRelativeImport=false, reportMissingImports=false, reportDeprecated=false
# pyright: reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportAny=false, reportImplicitStringConcatenation=false
# pyright: reportAssignmentType=false
# pyright: reportUnusedParameter=false
# pyright: reportCallIssue=false

from __future__ import annotations

import json
import logging
import time
from importlib import import_module
from typing import Protocol, Sequence, cast

from pydantic import ValidationError

logger = logging.getLogger(__name__)

from storage.persona_models import Persona
from podcast.prompts import format_persona_context, get_transcript_prompt
from podcast.llm_client import LLMConfig, LLMProvider, chat_completion, create_llm_client, get_default_config

_models_module = import_module("podcast.models")

TURN_TARGETS = {"short": 3, "medium": 6, "long": 10}


class _Speaker(Protocol):
    name: str
    voice_id: str
    role: str
    type: str


class _SpeakerProfile(Protocol):
    speakers: Sequence[_Speaker]


class _Segment(Protocol):
    title: str
    description: str
    size: str


class _Outline(Protocol):
    @property
    def segments(self) -> Sequence[_Segment]:
        ...


class _Dialogue(Protocol):
    speaker: str
    text: str


class _Transcript(Protocol):
    @property
    def dialogues(self) -> Sequence[_Dialogue]:
        ...

    @classmethod
    def model_validate(cls, data: object) -> "_Transcript":
        ...


OutlineClass = cast(type[_Outline], getattr(_models_module, "Outline"))
TranscriptClass = cast(type[_Transcript], getattr(_models_module, "Transcript"))
SpeakerProfileClass = cast(
    type[_SpeakerProfile], getattr(_models_module, "SpeakerProfile")
)


def _coerce_speakers(
    speakers: Sequence[_Speaker] | _SpeakerProfile,
) -> list[_Speaker]:
    if isinstance(speakers, SpeakerProfileClass):
        return list(speakers.speakers)
    return list(cast(Sequence[_Speaker], speakers))


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _speaker_name_map(speakers: Sequence[_Speaker]) -> dict[str, str]:
    name_map: dict[str, str] = {}
    for speaker in speakers:
        normalized = _normalize_name(speaker.name)
        name_map[normalized] = speaker.name
    return name_map


def _validate_speaker_names(dialogues: Sequence[_Dialogue], speakers: Sequence[_Speaker]) -> None:
    allowed = _speaker_name_map(speakers)
    for dialogue in dialogues:
        normalized = _normalize_name(dialogue.speaker)
        if normalized not in allowed:
            raise ValueError(f"Unexpected speaker name: {dialogue.speaker}")


def _canonicalize_speaker_names(
    dialogues: Sequence[_Dialogue], speakers: Sequence[_Speaker]
) -> list[dict[str, str]]:
    allowed = _speaker_name_map(speakers)
    canonicalized: list[dict[str, str]] = []
    for dialogue in dialogues:
        normalized = _normalize_name(dialogue.speaker)
        if normalized not in allowed:
            raise ValueError(f"Unexpected speaker name: {dialogue.speaker}")
        canonicalized.append(
            {"speaker": allowed[normalized], "text": dialogue.text}
        )
    return canonicalized


def _format_outline_for_prompt(outline: _Outline, topic: str) -> str:
    lines = [f"Topic: {topic}", "Segments:"]
    for index, segment in enumerate(outline.segments, start=1):
        lines.append(
            f"{index}. {segment.title} ({segment.size}) - {segment.description}"
        )
    return "\n".join(lines)


def _format_speaker_roles(speakers: Sequence[_Speaker]) -> str:
    if not speakers:
        return "- None provided"
    lines = []
    for speaker in speakers:
        lines.append(f"- {speaker.name}: {speaker.role}")
    return "\n".join(lines)


_format_persona_context = format_persona_context


def _segment_turns(size: str) -> int:
    normalized = size.strip().lower()
    return TURN_TARGETS.get(normalized, TURN_TARGETS["medium"])


def _format_segment_for_prompt(segment: _Segment) -> str:
    return (
        f"{segment.title} ({segment.size})\n"
        f"Summary: {segment.description}"
    )


def _parse_transcript_response(
    response_text: str, speakers: Sequence[_Speaker]
) -> list[dict[str, str]]:
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Invalid JSON response from LLM.") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    payload = cast(dict[str, object], payload)

    try:
        transcript = TranscriptClass.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("LLM response does not match Transcript schema.") from exc

    _validate_speaker_names(transcript.dialogues, speakers)
    return _canonicalize_speaker_names(transcript.dialogues, speakers)


def _fallback_dialogue(
    speakers: Sequence[_Speaker], segment: _Segment, is_final: bool
) -> list[dict[str, str]]:
    if not speakers:
        raise ValueError("At least one speaker is required for fallback dialogue.")
    host = next((speaker for speaker in speakers if speaker.role == "Host"), None)
    chosen = host or speakers[0]
    if is_final:
        text = (
            "We hit a technical snag on this segment, but that's all for today. "
            "Thanks for listening, and we'll see you next time."
        )
    else:
        text = (
            f"We ran into a technical snag on {segment.title}. "
            "Let's move to the next part of the conversation."
        )
    return [{"speaker": chosen.name, "text": text}]


def generate_transcript(
    outline: _Outline,
    topic: str,
    briefing: str,
    speakers: Sequence[_Speaker] | _SpeakerProfile,
    personas: dict[str, Persona] | None = None,
    language: str = "English",
    llm_config: LLMConfig | None = None,
) -> _Transcript:
    """Generate a podcast transcript using the configured LLM provider."""
    from openai import OpenAIError

    speaker_list = _coerce_speakers(speakers)
    if not speaker_list:
        raise ValueError("At least one speaker is required to generate a transcript.")

    outline_text = _format_outline_for_prompt(outline, topic)
    role_context = _format_speaker_roles(speaker_list)
    enriched_briefing = f"{briefing}\n\nSPEAKER ROLES:\n{role_context}"

    if personas:
        persona_context = _format_persona_context(personas)
        if persona_context:
            enriched_briefing += f"\n\nSPEAKER PERSONAS:\n{persona_context}"

    if llm_config is None:
        from config import get_openai_api_key
        llm_config = get_default_config(
            LLMProvider.OPENAI,
            api_key=get_openai_api_key(),
        )

    client = create_llm_client(llm_config)
    logger.info("Transcript generation: language=%s, provider=%s", language, llm_config.provider.value)

    dialogues: list[dict[str, str]] = []
    total_segments = len(outline.segments)

    for index, segment in enumerate(outline.segments):
        is_final = index == total_segments - 1
        turns = _segment_turns(segment.size)
        prompt = get_transcript_prompt(
            outline=outline_text,
            segment=_format_segment_for_prompt(segment),
            briefing=enriched_briefing,
            speakers=[speaker.name for speaker in speaker_list],
            is_final=is_final,
            turns=turns,
            language=language,
        )

        last_error: Exception | None = None
        segment_dialogues: list[dict[str, str]] | None = None
        for attempt in range(3):
            try:
                content = chat_completion(
                    client=client,
                    model=llm_config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You create podcast transcripts that are natural, "
                                "role-aware, and JSON-only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=llm_config.temperature,
                    json_mode=True,
                    provider=llm_config.provider,
                )
                segment_dialogues = _parse_transcript_response(content, speaker_list)
                break
            except (ValueError, OpenAIError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep((1, 2, 4)[attempt])

        if segment_dialogues is None:
            if last_error is not None:
                logger.warning("Segment '%s' failed after retries: %s", segment.title, last_error)
            segment_dialogues = _fallback_dialogue(speaker_list, segment, is_final)

        dialogues.extend(segment_dialogues)

    return TranscriptClass.model_validate({"dialogues": dialogues})
