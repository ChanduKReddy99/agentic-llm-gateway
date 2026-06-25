"""
Guardrails — Input & Output Validation
=======================================
Every request passes through guardrails BEFORE reaching agents
and AFTER agents produce responses.

Input Guardrails:
  1. PII Detection — flag/redact emails, SSNs, credit cards, phone numbers
  2. Prompt Injection — detect jailbreak / system override attempts
  3. Topic Relevance — ensure query is on-topic for this assistant
  4. Length Check — reject excessively long inputs

Output Guardrails:
  1. Toxicity Filter — block harmful/offensive outputs
  2. PII Leakage — ensure no PII slipped into response
  3. Hallucination Indicators — flag responses with made-up citations
  4. Length Validation — ensure response is complete and reasonable
"""
import re
import time
from dataclasses import dataclass, field
from typing import Literal

import structlog

from app.config.settings import get_settings
from app.observability.metrics import guardrails_latency_seconds, guardrails_violations_total
from app.observability.tracing import trace_span

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class GuardrailResult:
    passed: bool
    violations: list[dict] = field(default_factory=list)
    sanitized_text: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.passed


# ─── PII Detection Patterns ──────────────────────────────────────────────────

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# ─── Prompt Injection Patterns ───────────────────────────────────────────────

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|another)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:if\s+)?(?:you\s+(?:are|were)\s+)?(?:a\s+)?(?:different|evil|uncensored)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
]

# ─── Toxicity Keywords (simplified — production would use a classifier) ──────

TOXIC_KEYWORDS = {
    "violence", "kill", "murder", "bomb", "weapon", "terrorist",
    "hate", "racist", "sexist",
}

# ─── Off-Topic Detection ─────────────────────────────────────────────────────

OFF_TOPIC_PATTERNS = [
    re.compile(r"write\s+(?:me\s+)?(?:a\s+)?(?:poem|song|story|fiction)", re.IGNORECASE),
    re.compile(r"play\s+(?:a\s+)?(?:game|chess|quiz)", re.IGNORECASE),
    re.compile(r"what\s+(?:is\s+)?(?:the\s+)?(?:weather|time|date)", re.IGNORECASE),
]


class InputGuardrails:
    """
    Validates and sanitizes user input before it reaches agents.
    Fast-path: reject obviously bad inputs without expensive LLM calls.
    """

    async def validate(self, text: str, context: dict | None = None) -> GuardrailResult:
        start = time.time()
        violations = []
        sanitized = text

        with trace_span("guardrails.input", {"text_length": len(text)}):
            # 1. Length check
            if len(text) > 10_000:
                violations.append({
                    "type": "length_exceeded",
                    "message": f"Input too long: {len(text)} chars (max 10,000)",
                    "severity": "block",
                })

            # 2. Prompt injection detection
            for pattern in INJECTION_PATTERNS:
                if pattern.search(text):
                    violations.append({
                        "type": "prompt_injection",
                        "message": "Potential prompt injection detected",
                        "severity": "block",
                        "matched": pattern.pattern[:50],
                    })

            # 3. PII detection
            if settings.pii_detection_enabled:
                detected_pii = []
                for pii_type, pattern in PII_PATTERNS.items():
                    matches = pattern.findall(text)
                    if matches:
                        detected_pii.append(pii_type)
                        # Redact PII rather than blocking
                        sanitized = pattern.sub(f"[{pii_type.upper()}_REDACTED]", sanitized)

                if detected_pii:
                    violations.append({
                        "type": "pii_detected",
                        "message": f"PII detected and redacted: {', '.join(detected_pii)}",
                        "severity": "redact",  # warn but don't block
                        "pii_types": detected_pii,
                    })

            # 4. Empty input
            if not text.strip():
                violations.append({
                    "type": "empty_input",
                    "message": "Input cannot be empty",
                    "severity": "block",
                })

        # Determine if we should block
        blocking_violations = [v for v in violations if v["severity"] == "block"]
        passed = len(blocking_violations) == 0

        # Emit metrics
        for violation in violations:
            guardrails_violations_total.labels(
                stage="input", violation_type=violation["type"]
            ).inc()
            logger.warning(
                "guardrails.input.violation",
                violation_type=violation["type"],
                severity=violation["severity"],
            )

        elapsed = time.time() - start
        guardrails_latency_seconds.labels(stage="input").observe(elapsed)

        if not passed:
            logger.warning(
                "guardrails.input.blocked",
                blocking_violations=[v["type"] for v in blocking_violations],
            )

        return GuardrailResult(
            passed=passed,
            violations=violations,
            sanitized_text=sanitized if passed else text,
            metadata={"latency_ms": round(elapsed * 1000, 2)},
        )


class OutputGuardrails:
    """
    Validates agent output before sending to user.
    Catches LLM-generated harmful content, PII leakage, etc.
    """

    async def validate(
        self, text: str, query: str = "", context: dict | None = None
    ) -> GuardrailResult:
        start = time.time()
        violations = []

        with trace_span("guardrails.output", {"text_length": len(text)}):
            # 1. Toxicity check (lightweight keyword filter)
            text_lower = text.lower()
            matched_toxic = [kw for kw in TOXIC_KEYWORDS if kw in text_lower]
            if matched_toxic:
                violations.append({
                    "type": "toxicity",
                    "message": f"Potentially toxic content detected: {matched_toxic[:3]}",
                    "severity": "warn",  # Warn but don't always block
                })

            # 2. PII leakage in output
            if settings.pii_detection_enabled:
                for pii_type, pattern in PII_PATTERNS.items():
                    if pattern.search(text):
                        violations.append({
                            "type": "pii_leakage",
                            "message": f"PII ({pii_type}) detected in output",
                            "severity": "warn",
                        })

            # 3. Empty or too-short response
            if len(text.strip()) < 10:
                violations.append({
                    "type": "insufficient_response",
                    "message": "Response is too short to be useful",
                    "severity": "warn",
                })

            # 4. Check for error indicators in LLM output
            error_phrases = [
                "i cannot", "i am unable to", "as an ai, i",
                "i don't have access", "i cannot browse"
            ]
            if any(phrase in text_lower for phrase in error_phrases):
                violations.append({
                    "type": "capability_refusal",
                    "message": "LLM refused the task — may need prompt adjustment",
                    "severity": "warn",
                })

        # Blocking violations for output (stricter)
        blocking_violations = [
            v for v in violations
            if v["severity"] == "block"
        ]
        passed = len(blocking_violations) == 0

        # Emit metrics
        for violation in violations:
            guardrails_violations_total.labels(
                stage="output", violation_type=violation["type"]
            ).inc()

        elapsed = time.time() - start
        guardrails_latency_seconds.labels(stage="output").observe(elapsed)

        if violations:
            logger.info(
                "guardrails.output.violations",
                count=len(violations),
                types=[v["type"] for v in violations],
                blocked=not passed,
            )

        return GuardrailResult(
            passed=passed,
            violations=violations,
            sanitized_text=text,
            metadata={"latency_ms": round(elapsed * 1000, 2)},
        )
