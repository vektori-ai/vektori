"""Write-time secret scanning — refuse to memorize credentials.

Pattern + entropy heuristics. Deliberately conservative: false positives are
recoverable (user rewrites the note); leaked keys in a synced memory dir are not.
"""

from __future__ import annotations

import math
import re

_PATTERNS = [
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("openai-key", re.compile(r"sk-[A-Za-z0-9_-]{32,}")),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_-]{32,}")),
    ("google-api-key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("generic-assignment", re.compile(
        r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\x22]?([A-Za-z0-9_\-/+]{20,})")),
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def scan_text(text: str) -> list[dict]:
    """Returns findings: [{rule, match}] — empty list means clean."""
    findings: list[dict] = []
    for rule, pat in _PATTERNS:
        for m in pat.finditer(text):
            token = m.group(1) if m.groups() else m.group(0)
            if rule == "generic-assignment" and _entropy(token) < 3.5:
                continue  # low-entropy values like "password: changeme-later"
            findings.append({"rule": rule, "match": token[:8] + "…"})
    return findings


class SecretsFoundError(ValueError):
    def __init__(self, findings: list[dict]):
        rules = ", ".join(sorted({f["rule"] for f in findings}))
        super().__init__(
            "refusing to store memory containing probable secrets (" + rules + "); "
            "redact them or pass secret_scan=False"
        )
        self.findings = findings
