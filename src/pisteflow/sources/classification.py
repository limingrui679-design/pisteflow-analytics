"""Conservative rule-based labels for heterogeneous fencing catalog metadata."""

from __future__ import annotations

import re

from pisteflow.models import ContentKind, Weapon

_WORD_PATTERNS = {
    Weapon.EPEE: re.compile(r"\b(?:epee|épée|espada|spada|degen)\b", re.IGNORECASE),
    Weapon.FOIL: re.compile(r"\b(?:foil|fleuret|florete|fioretto|florett)\b", re.IGNORECASE),
    Weapon.SABRE: re.compile(r"\b(?:sabre|saber|sable|sciabola|säbel|sabel)\b", re.IGNORECASE),
}

_UPPERCASE_ABBREVIATIONS = {
    Weapon.EPEE: re.compile(r"(?<![A-Z])(?:ME|WE|MEE|WEE|MIE|WIE|MTE|WTE)(?![A-Z])"),
    Weapon.FOIL: re.compile(r"(?<![A-Z])(?:MF|WF|MFO|WFO|MIF|WIF|MTF|WTF)(?![A-Z])"),
    Weapon.SABRE: re.compile(r"(?<![A-Z])(?:MS|WS|MSA|WSA|MIS|WIS|MTS|WTS)(?![A-Z])"),
}
_LEGACY_DATED_CODE = re.compile(r"(?:^|\b\d{6,8})[ _-]+([mw])([efs])(?=[ _-])", re.IGNORECASE)
_SPACED_EVENT_CODE = re.compile(
    r"\b([MW])\s+([EFS])\s+(?:Individual|Team)\b",
    re.IGNORECASE,
)
_REVERSED_EVENT_CODE = re.compile(
    r"\b([EFS])([MW])(?=\s+(?:Individual|Team|champion|final))",
    re.IGNORECASE,
)
_CODE_TO_WEAPON = {"e": Weapon.EPEE, "f": Weapon.FOIL, "s": Weapon.SABRE}

_BOUT_PATTERN = re.compile(
    r"(?:\bvs\.?\b|\bversus\b|\bbout\s*#?\d+\b|\b(?:t|l)\s*(?:8|16|32|64|128)\b)",
    re.IGNORECASE,
)
_STREAM_PATTERN = re.compile(
    r"(?:\bpiste\b|\bpool(?:s)?\b|\bround\b|\bsemi[- ]?finals?\b|\bfinals?\b|"
    r"\bworld cup\b|\bgrand prix\b|\bchampionships?\b|\bteam(?:s)?\b)",
    re.IGNORECASE,
)
_HIGHLIGHT_PATTERN = re.compile(
    r"(?:\bhighlights?\b|\bbest (?:touches|moments)\b|\btop \d+\b|world champion)",
    re.IGNORECASE,
)
_INSTRUCTION_PATTERN = re.compile(
    r"(?:\bhow to\b|\btraining\b|\bdrills?\b|\blessons?\b|\btechniques?\b|\bfootwork\b|"
    r"\bvisual dictionary\b|\bEVD[-_])",
    re.IGNORECASE,
)
_INTERVIEW_PATTERN = re.compile(
    r"(?:\binterviews?\b|\bconference\b|\bpress conference\b|\bpodcasts?\b|"
    r"\bclosing ceremony\b|\bopening ceremony\b|\binside fencing\b)",
    re.IGNORECASE,
)


def infer_weapon_candidates(*texts: str | None) -> tuple[Weapon, ...]:
    """Return every weapon explicitly supported by the supplied text.

    Short abbreviations are intentionally case-sensitive: lower-case English
    words such as ``we`` and ``me`` must never become weapon labels.
    """

    candidates: set[Weapon] = set()
    for text in texts:
        if not text:
            continue
        normalized = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
        for weapon, pattern in _WORD_PATTERNS.items():
            if pattern.search(normalized):
                candidates.add(weapon)
        for weapon, pattern in _UPPERCASE_ABBREVIATIONS.items():
            if pattern.search(normalized):
                candidates.add(weapon)
        for match in _LEGACY_DATED_CODE.finditer(normalized):
            candidates.add(_CODE_TO_WEAPON[match.group(2).lower()])
        for match in _SPACED_EVENT_CODE.finditer(normalized):
            candidates.add(_CODE_TO_WEAPON[match.group(2).lower()])
        for match in _REVERSED_EVENT_CODE.finditer(normalized):
            candidates.add(_CODE_TO_WEAPON[match.group(1).lower()])
    return tuple(sorted(candidates, key=lambda weapon: weapon.value))


def primary_weapon(candidates: tuple[Weapon, ...]) -> Weapon:
    """Use a weapon label only when the metadata identifies exactly one."""

    return candidates[0] if len(candidates) == 1 else Weapon.UNKNOWN


def classify_content(
    title: str,
    *,
    description: str | None = None,
    duration_seconds: float | None = None,
    athlete_count: int = 0,
) -> ContentKind:
    """Classify catalog content without claiming frame-level review."""

    text = " ".join(part for part in (title, description) if part)
    if _INTERVIEW_PATTERN.search(text):
        return "interview"
    if _INSTRUCTION_PATTERN.search(text):
        return "instructional"
    if _HIGHLIGHT_PATTERN.search(text):
        return "highlight"
    if _BOUT_PATTERN.search(text) or (athlete_count >= 2 and "bout" in text.lower()):
        return "bout"
    if _STREAM_PATTERN.search(text):
        return "multi_bout_stream"
    if duration_seconds is not None and duration_seconds <= 90:
        return "highlight"
    return "other"
