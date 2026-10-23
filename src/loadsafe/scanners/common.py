"""Helpers shared by scanners."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..findings import Finding, Severity
from ..policy import Policy


@dataclass
class Ctx:
    """Per-artifact scan context collecting findings and metadata."""

    path: str
    policy: Policy
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(
        self, rule: str, sev: Severity, msg: str, location: str | None = None, **details: Any
    ) -> None:
        self.findings.append(Finding(rule, sev, msg, self.path, location, details))


# Template-injection markers for Jinja chat templates (GGUF / tokenizer_config.json).
SSTI_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(rx))
    for name, rx in [
        (
            "dunder attribute",
            r"__(class|globals|builtins|subclasses|mro|base|bases|init|import"
            r"|getattribute|reduce|reduce_ex|code|dict|func|self|loader)__",
        ),
        ("attr filter with dunder", r"\|\s*attr\s*\(\s*['\"][^'\"]*(_|\\x5f)"),
        ("hex/unicode escaped underscore", r"\\x5f|\\u005f|\\137"),
        ("os/subprocess reference", r"\b(os|subprocess)\s*\.\s*\w+|\bpopen\s*\("),
        ("import/eval/exec call", r"\b(import|eval|exec|compile|getattr)\s*\("),
        ("jinja global gadget", r"\b(cycler|joiner|lipsum|namespace|self)\s*\.\s*_"),
        (
            "request/config object",
            r"\b(request|config|url_for|get_flashed_messages)\s*\.\s*"
            r"(application|__|from_|items)",
        ),
    ]
]


def ssti_hits(template: str) -> list[str]:
    return [name for name, rx in SSTI_PATTERNS if rx.search(template)]


CODE_LIKE = re.compile(
    r"(__import__|\bexec\s*\(|\beval\s*\(|os\.system|subprocess|<script|\bimport\s+\w+|"
    r"\{\{.*\}\}|\{%.*%\}|base64\.b64decode|/bin/(ba)?sh|powershell|curl\s+http|wget\s+http)",
    re.IGNORECASE | re.DOTALL,
)
URL_RE = re.compile(r"\b(https?|ftp|file)://[^\s\"']+", re.IGNORECASE)
LONG_B64 = re.compile(r"[A-Za-z0-9+/]{512,}={0,2}")

EXECUTABLE_MAGICS: list[tuple[bytes, str]] = [
    (b"\x7fELF", "ELF executable/shared object"),
    (b"MZ", "Windows PE executable"),
    (b"\xfe\xed\xfa\xce", "Mach-O executable"),
    (b"\xfe\xed\xfa\xcf", "Mach-O executable"),
    (b"\xce\xfa\xed\xfe", "Mach-O executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable"),
    (b"\xca\xfe\xba\xbe", "Mach-O universal binary / Java class"),
    (b"#!", "script with shebang"),
]


def executable_kind(head: bytes) -> str | None:
    for magic, kind in EXECUTABLE_MAGICS:
        if head.startswith(magic):
            if magic == b"MZ" and len(head) < 64:
                return None
            return kind
    return None
