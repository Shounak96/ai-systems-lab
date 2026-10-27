"""Deterministic query-side guards for correctness-sensitive attributes.

The guards are deliberately conservative: a mismatch means "do not reuse", not
"the queries are proven non-equivalent". They provide no adversarial guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class GuardConfig:
    numbers: bool = True
    dates: bool = True
    negation: bool = True
    versions: bool = True
    entities: bool = True


@dataclass(frozen=True)
class GuardDecision:
    accepted: bool
    reasons: tuple[str, ...]


_NUMBER_RE = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?%?(?!\w)")
_ISO_DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")
_MONTH_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+(?:19|20)\d{2})?\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_VERSION_RE = re.compile(
    r"\b(?:(?:v(?:ersion)?\s*)(\d+(?:\.\d+){0,3})|(\d+(?:\.\d+){1,3}))"
    r"(?:[-_]?([a-z]+\d*))?\b",
    re.IGNORECASE,
)
_CONTEXT_VERSION_RE = re.compile(
    r"\b(?:api|python|postgresql|node\.js|release)\s+(?:version\s+|v)?"
    r"(\d+(?:\.\d+){0,3})\b",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"\b(?:no|not|never|without|cannot|can't|disable|disabled|deny|denied|"
    r"disallow|unavailable|down|offline|turn\s+off|prevent|forbid|forbidden)\b",
    re.IGNORECASE,
)
_POSITIVE_RE = re.compile(
    r"\b(?:enable|enabled|allow|allowed|available|up|online|turn\s+on|permit|permitted)\b",
    re.IGNORECASE,
)

# A deliberately finite catalog: only reliably recognized product/platform names
# participate. Unknown nouns are left to the embedding threshold.
ENTITY_ALIASES = {
    **{name: name for name in (
        "aws", "azure", "cloudflare", "docker", "dynamodb", "fastapi", "github",
        "gitlab", "grafana", "java", "kafka", "kubernetes", "linux", "mongodb",
        "mysql", "nginx", "node.js", "openai", "oracle", "postgresql", "prometheus",
        "python", "rabbitmq", "redis", "slack", "terraform", "windows",
    )},
    "amazon s3": "amazon s3",
    "s3": "amazon s3",
}


def _numbers(text: str) -> tuple[str, ...]:
    # Dates and versions have dedicated guards. Masking them here prevents a
    # notation-only change such as "v2.0" vs "version 2.0" from looking numeric.
    masked = text
    for pattern in (_ISO_DATE_RE, _MONTH_DATE_RE, _VERSION_RE, _CONTEXT_VERSION_RE):
        masked = pattern.sub(" ", masked)
    return tuple(sorted(match.group(0).replace(",", "").lower() for match in _NUMBER_RE.finditer(masked)))


def _dates(text: str) -> tuple[str, ...]:
    values = [match.group(0).lower().replace(",", "") for match in _ISO_DATE_RE.finditer(text)]
    values.extend(match.group(0).lower().replace(",", "") for match in _MONTH_DATE_RE.finditer(text))
    values.extend(match.group(0) for match in _YEAR_RE.finditer(text))
    return tuple(sorted(set(values)))


def _versions(text: str) -> tuple[str, ...]:
    values = []
    for match in _VERSION_RE.finditer(text):
        raw = match.group(1) or match.group(2)
        parts = [str(int(part)) for part in raw.split(".")]
        while len(parts) > 1 and parts[-1] == "0":
            parts.pop()
        normalized = ".".join(parts)
        if match.group(3):
            normalized += f"-{match.group(3).lower()}"
        values.append(normalized)
    for match in _CONTEXT_VERSION_RE.finditer(text):
        parts = [str(int(part)) for part in match.group(1).split(".")]
        while len(parts) > 1 and parts[-1] == "0":
            parts.pop()
        values.append(".".join(parts))
    return tuple(sorted(set(values)))


def _polarity(text: str) -> str:
    negative = bool(_NEGATIVE_RE.search(text))
    positive = bool(_POSITIVE_RE.search(text))
    if negative and not positive:
        return "negative"
    if positive and not negative:
        return "positive"
    if negative and positive:
        return "mixed"
    return "unmarked"


def _entities(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    found = []
    for entity, canonical in ENTITY_ALIASES.items():
        if re.search(rf"(?<![\w.]){re.escape(entity)}(?![\w.])", lowered):
            found.append(canonical)
    return tuple(sorted(set(found)))


def compare_sensitive_attributes(
    incoming: str,
    cached: str,
    config: GuardConfig = GuardConfig(),
) -> GuardDecision:
    """Reject attribute mismatches; return auditable reason codes."""
    reasons: list[str] = []
    if config.numbers and _numbers(incoming) != _numbers(cached):
        reasons.append("number_mismatch")
    if config.dates and _dates(incoming) != _dates(cached):
        reasons.append("date_mismatch")
    if config.versions and _versions(incoming) != _versions(cached):
        reasons.append("version_mismatch")
    if config.negation:
        incoming_polarity, cached_polarity = _polarity(incoming), _polarity(cached)
        if incoming_polarity != cached_polarity and "unmarked" not in {incoming_polarity, cached_polarity}:
            reasons.append("polarity_mismatch")
        elif bool(_NEGATIVE_RE.search(incoming)) != bool(_NEGATIVE_RE.search(cached)):
            reasons.append("negation_mismatch")
    if config.entities:
        incoming_entities, cached_entities = _entities(incoming), _entities(cached)
        if incoming_entities != cached_entities and (incoming_entities or cached_entities):
            reasons.append("entity_mismatch")
    return GuardDecision(accepted=not reasons, reasons=tuple(reasons))
