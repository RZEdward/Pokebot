"""Typed high-level command intents; this module does not issue motor commands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class IntentKind(StrEnum):
    STOP = "stop"
    STAND = "stand"
    WALK_FORWARD = "walk_forward"
    WALK_BACKWARD = "walk_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    UNKNOWN = "unknown"


class DispatchStatus(StrEnum):
    ACCEPTED = "accepted"
    UNKNOWN_INTENT = "unknown_intent"
    LOW_CONFIDENCE = "low_confidence"
    SKILL_UNAVAILABLE = "skill_unavailable"


@dataclass(frozen=True, slots=True)
class SkillIntent:
    kind: IntentKind
    source: str
    raw_text: str
    confidence: float
    supported_by_demo: bool


@dataclass(frozen=True, slots=True)
class SkillDispatch:
    status: DispatchStatus
    intent: SkillIntent
    skill: str | None
    reason: str


_PHRASES: tuple[tuple[IntentKind, tuple[str, ...]], ...] = (
    (IntentKind.STOP, ("stop", "halt", "freeze", "emergency stop")),
    (IntentKind.STAND, ("stand", "stand up", "hold still", "balance")),
    (IntentKind.WALK_FORWARD, ("walk forward", "move forward", "go forward")),
    (IntentKind.WALK_BACKWARD, ("walk backward", "move backward", "go backward")),
    (IntentKind.TURN_LEFT, ("turn left", "rotate left")),
    (IntentKind.TURN_RIGHT, ("turn right", "rotate right")),
)


def parse_transcript(text: str, *, confidence: float = 1.0, source: str = "text") -> SkillIntent:
    """Map ASR text into a small allowlisted skill-intent vocabulary.

    This is the post-ASR parser boundary. It deliberately does not call an LLM,
    choose joint targets, or claim that any locomotion skill is implemented.
    STOP is accepted as an intent but a physical e-stop must remain independent.
    """
    if not isfinite(confidence):
        raise ValueError("Transcript confidence must be finite")
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    normalized = " ".join(normalized.split())
    if re.search(r"\b(?:not|never|cannot)\b|\bdon\s+t\b|\bcan\s+t\b", normalized):
        return SkillIntent(
            kind=IntentKind.UNKNOWN,
            source=source,
            raw_text=text,
            confidence=max(0.0, min(1.0, confidence)),
            supported_by_demo=False,
        )
    for kind, phrases in _PHRASES:
        for phrase in phrases:
            match = re.search(rf"\b{re.escape(phrase)}\b", normalized)
            if match is None:
                continue
            return SkillIntent(
                kind=kind,
                source=source,
                raw_text=text,
                confidence=max(0.0, min(1.0, confidence)),
                supported_by_demo=kind in (IntentKind.STOP, IntentKind.STAND),
            )
    return SkillIntent(
        kind=IntentKind.UNKNOWN,
        source=source,
        raw_text=text,
        confidence=max(0.0, min(1.0, confidence)),
        supported_by_demo=False,
    )


def dispatch_intent(
    intent: SkillIntent,
    *,
    available_skills: frozenset[str] = frozenset({"balance"}),
    minimum_confidence: float = 0.75,
) -> SkillDispatch:
    """Resolve a typed intent to a registered skill request, never to joints.

    STOP requests a supervisor safe-stop event. A real electrical emergency
    stop remains an independent circuit and must not depend on this function.
    """
    if not isfinite(minimum_confidence) or not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be finite and within [0, 1]")
    if intent.kind is IntentKind.UNKNOWN:
        return SkillDispatch(
            DispatchStatus.UNKNOWN_INTENT, intent, None, "No allowlisted intent matched."
        )
    if intent.confidence < minimum_confidence:
        return SkillDispatch(
            DispatchStatus.LOW_CONFIDENCE, intent, None, "Transcript confidence is below threshold."
        )

    if intent.kind is IntentKind.STOP:
        return SkillDispatch(
            DispatchStatus.ACCEPTED, intent, "supervisor.safe_stop_request", "Request safe stop."
        )

    skill_by_intent = {
        IntentKind.STAND: "balance",
        IntentKind.WALK_FORWARD: "walk_forward",
        IntentKind.WALK_BACKWARD: "walk_backward",
        IntentKind.TURN_LEFT: "turn_left",
        IntentKind.TURN_RIGHT: "turn_right",
    }
    skill = skill_by_intent[intent.kind]
    if skill not in available_skills:
        return SkillDispatch(
            DispatchStatus.SKILL_UNAVAILABLE, intent, None, f"Skill '{skill}' is not registered."
        )
    return SkillDispatch(DispatchStatus.ACCEPTED, intent, skill, f"Request skill '{skill}'.")
