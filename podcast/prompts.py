"""Prompt templates for podcast outline and transcript generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from storage.persona_models import Persona


def format_persona_context(personas: dict[str, Persona] | None) -> str:
    if not personas:
        return ""
    lines = []
    for _, persona in personas.items():
        full_context = f"{persona.background} {persona.bio}".strip()
        context_truncated = full_context[:200] + "..." if len(full_context) > 200 else full_context
        expertise_str = ", ".join(persona.expertise[:3]) if persona.expertise else "General"
        line = (
            f"- {persona.character_name}: {persona.personality}, {persona.speaking_style} speaker. "
            f"Expertise: {expertise_str}. {context_truncated}"
        )
        lines.append(line)
    return "\n".join(lines)


def get_transcript_prompt(
    outline: str,
    segment: str,
    briefing: str,
    speakers: list[str],
    is_final: bool = False,
    turns: int = 8,
    language: str = "English",
) -> str:
    dialogue_schema = {
        "type": "object",
        "properties": {
            "dialogues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {
                            "type": "string",
                            "description": "Name of the speaker",
                        },
                        "text": {
                            "type": "string",
                            "description": "Spoken dialogue (natural, conversational)",
                        },
                    },
                    "required": ["speaker", "text"],
                },
                "minItems": 2,
                "maxItems": 50,
            }
        },
        "required": ["dialogues"],
    }

    speaker_roles = {
        "Host": "Asks questions, guides conversation, keeps things on track, shows enthusiasm",
        "Expert": "Provides detailed explanations, shares knowledge, answers questions thoroughly",
        "Guest": "Shares personal experience, offers perspective, responds naturally to questions",
        "Narrator": "Provides context and transitions between topics",
    }

    examples = """
EXAMPLE 1: Expert Interview Segment
Segment: "Current State of AI in Diagnosis"
Speakers: ["Host", "Dr. Smith (Expert)"]

Output:
{
  "dialogues": [
    {
      "speaker": "Host",
      "text": "Dr. Smith, let's dive into how AI is actually being used in medical diagnosis today. Can you give us some concrete examples?"
    },
    {
      "speaker": "Dr. Smith",
      "text": "Absolutely. One of the most successful applications is in radiology. AI systems can now detect certain cancers in X-rays and CT scans with accuracy rates that match or exceed human radiologists."
    },
    {
      "speaker": "Host",
      "text": "That's impressive. Are there other areas where you're seeing significant impact?"
    },
    {
      "speaker": "Dr. Smith",
      "text": "Yes, pathology is another big one. AI can analyze tissue samples to identify diseases like cancer much faster than traditional methods. We're also seeing promising results in cardiology and ophthalmology."
    },
    {
      "speaker": "Host",
      "text": "How long does it typically take to implement these systems in a hospital?"
    },
    {
      "speaker": "Dr. Smith",
      "text": "That varies, but usually between 6 to 18 months. It depends on the complexity of the system and how well it integrates with existing infrastructure."
    }
  ]
}

EXAMPLE 2: Panel Discussion Segment
Segment: "Work-Life Balance Challenges"
Speakers: ["Host", "Manager", "Employee"]

Output:
{
  "dialogues": [
    {
      "speaker": "Host",
      "text": "One thing we hear a lot about with remote work is the challenge of maintaining work-life balance. Manager, from your perspective, how are you seeing this play out?"
    },
    {
      "speaker": "Manager",
      "text": "It's definitely a concern. Without the physical separation of an office, people tend to work longer hours. We've had to be intentional about encouraging breaks and respecting off-hours."
    },
    {
      "speaker": "Employee",
      "text": "I can relate to that. At first, I found myself working until 8 or 9 PM because my home office was just steps away. I had to set hard boundaries."
    },
    {
      "speaker": "Host",
      "text": "What kind of boundaries did you set?"
    },
    {
      "speaker": "Employee",
      "text": "I created a ritual where I literally close my laptop and put it away at 5 PM. I also changed my Slack status to 'offline' so people know not to expect immediate responses."
    },
    {
      "speaker": "Manager",
      "text": "That's smart. We've also implemented 'no meeting Fridays' in the afternoon to give people time to catch up and decompress."
    }
  ]
}
"""

    prompt = f"""You are an expert podcast scriptwriter creating natural, engaging dialogue for a podcast segment.

IMPORTANT: Generate all dialogue in {language}. The entire transcript must be written in {language}.

PODCAST OUTLINE (for context):
{outline}

CURRENT SEGMENT TO WRITE:
{segment}

BACKGROUND & BRIEFING:
{briefing}

SPEAKERS:
{", ".join(speakers)}

SPEAKER ROLE GUIDELINES:
{chr(10).join(f"- {name}: {role}" for name, role in speaker_roles.items())}

REQUIREMENTS:
1. Generate approximately {turns} dialogue turns (back-and-forth exchanges).
2. Write ALL dialogue in {language} - this is mandatory.
3. Make dialogue natural and conversational, not scripted or robotic.
4. Include follow-up questions from the Host to keep conversation flowing.
5. Vary sentence length and structure for natural pacing.
6. Use contractions and natural speech patterns appropriate for {language}.
7. Avoid jargon unless the Expert is explaining it.
8. Include natural transitions and acknowledgments appropriate for {language}.
9. {"Focus on concluding remarks, key takeaways, and wrapping up the discussion." if is_final else "Keep the discussion engaging and exploratory."}
10. Ensure all speakers get roughly equal speaking time.
11. Make the dialogue sound like a real conversation, not a Q&A interview.
12. Use speaker names exactly as listed in SPEAKERS; do not output role labels unless the role text is an exact speaker name.

OUTPUT FORMAT:
Return ONLY valid JSON matching this schema:
{json.dumps(dialogue_schema, indent=2)}

EXAMPLES OF GOOD DIALOGUE:
{examples}

Now create the transcript for this segment:"""

    return prompt


if __name__ == "__main__":
    transcript_prompt = get_transcript_prompt(
        outline="1. Introduction\n2. Main Discussion\n3. Closing",
        segment="Main Discussion: Productivity Trends",
        briefing="Focus on measurable productivity metrics and employee feedback.",
        speakers=["Host", "HR Manager"],
        is_final=False,
        turns=6,
    )

    print("=" * 80)
    print("TRANSCRIPT PROMPT TEST")
    print("=" * 80)
    print(transcript_prompt[:500] + "...\n")
    assert "Main Discussion" in transcript_prompt
    assert "dialogues" in transcript_prompt
    assert "Host" in transcript_prompt
    assert "HR Manager" in transcript_prompt
    print("✓ Transcript prompt renders correctly\n")

    # Test final segment variant
    final_transcript_prompt = get_transcript_prompt(
        outline="1. Introduction\n2. Main Discussion\n3. Closing",
        segment="Closing: Key Takeaways",
        briefing="Summarize main points and provide actionable insights.",
        speakers=["Host", "HR Manager"],
        is_final=True,
        turns=4,
    )

    print("=" * 80)
    print("FINAL SEGMENT TRANSCRIPT PROMPT TEST")
    print("=" * 80)
    print(final_transcript_prompt[:500] + "...\n")
    assert "concluding remarks" in final_transcript_prompt
    assert "Key Takeaways" in final_transcript_prompt
    print("✓ Final segment prompt renders correctly\n")

    print("=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
