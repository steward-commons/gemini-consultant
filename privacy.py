#!/usr/bin/env python3
"""Privacy guard for outbound Gemini calls.

Refuses to send text that matches a configurable denylist (plus a small
built-in set) unless the caller explicitly passes --i-have-owner-ok.
This does not modify the original text; it only inspects a copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Always-on markers, even if the denylist file is missing or empty.
BUILTIN_LITERALS: tuple[str, ...] = (
    "PRIVATE",
    "vault",
    "password",
    "~/.config",
    "Private - Sensitive Records",
)
BUILTIN_REGEX: tuple[str, ...] = (
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-shaped
)


@dataclass(frozen=True)
class PrivacyHit:
    label: str
    kind: str  # "literal" or "regex"


class PrivacyBlocked(Exception):
    """Raised when outbound text looks private and no waiver was given."""

    def __init__(self, hits: list[PrivacyHit]) -> None:
        self.hits = hits
        labels = ", ".join(sorted({hit.label for hit in hits}))
        super().__init__(
            "Privacy guard refused this request. It looks like private or "
            f"sensitive text ({labels}). Pass --i-have-owner-ok only if the owner "
            "explicitly said this content may go to Gemini."
        )


def load_denylist(
    path: Path | None,
) -> tuple[list[str], list[tuple[str, re.Pattern[str]]]]:
    literals = list(BUILTIN_LITERALS)
    regexes: list[tuple[str, re.Pattern[str]]] = [
        (pattern, re.compile(pattern, re.IGNORECASE)) for pattern in BUILTIN_REGEX
    ]
    if path is None or not path.is_file():
        return literals, regexes
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("re:"):
            pattern = line[3:].strip()
            if pattern:
                regexes.append((pattern, re.compile(pattern, re.IGNORECASE)))
        else:
            literals.append(line)
    return literals, regexes


def scan_text(text: str, path: Path | None = None) -> list[PrivacyHit]:
    if not text:
        return []
    literals, regexes = load_denylist(path)
    hits: list[PrivacyHit] = []
    seen: set[tuple[str, str]] = set()
    lowered = text.lower()
    for literal in literals:
        if literal.lower() in lowered:
            key = ("literal", literal.lower())
            if key not in seen:
                seen.add(key)
                hits.append(PrivacyHit(label=literal, kind="literal"))
    for label, pattern in regexes:
        if pattern.search(text):
            key = ("regex", label)
            if key not in seen:
                seen.add(key)
                hits.append(PrivacyHit(label=label, kind="regex"))
    return hits


def assert_sendable(
    texts: list[str],
    *,
    denylist: Path | None,
    waived: bool,
) -> None:
    """Raise PrivacyBlocked if any text hits the denylist and no waiver was given."""
    if waived:
        return
    hits: list[PrivacyHit] = []
    seen: set[tuple[str, str]] = set()
    for text in texts:
        for hit in scan_text(text or "", denylist):
            key = (hit.kind, hit.label)
            if key not in seen:
                seen.add(key)
                hits.append(hit)
    if hits:
        raise PrivacyBlocked(hits)
