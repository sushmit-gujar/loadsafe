"""SARIF 2.1.0 output for GitHub code scanning."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .. import __version__
from ..findings import Finding, ScanReport, Severity
from ..rules import ATLAS_NAMES, RULES

_LEVEL = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
    Severity.INFO: "note",
}
_SCORE = {
    Severity.CRITICAL: "9.5",
    Severity.HIGH: "8.0",
    Severity.MEDIUM: "5.5",
    Severity.LOW: "3.0",
    Severity.INFO: "1.0",
}
_LINE = re.compile(r"^line (\d+)$")
REPO_URL = "https://github.com/sushmit-gujar/loadsafe"


def _uri(path: str) -> str:
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/") if not re.match(r"^[a-z]+://", p) else p


def to_sarif(report: ScanReport, **_: object) -> str:
    findings = report.all_findings
    worst: dict[str, Severity] = {}
    for f in findings:
        worst[f.rule_id] = max(worst.get(f.rule_id, Severity.INFO), f.severity)
    rule_ids = sorted(worst)
    rules: list[dict[str, Any]] = []
    for rid in rule_ids:
        r = RULES[rid]
        tags = [
            "security",
            *r.owasp_llm,
            *(f"{t} {ATLAS_NAMES.get(t, '')}".strip() for t in r.atlas),
            *(f"external/cwe/{c.lower()}" for c in r.cwe),
        ]
        md = (
            f"{r.description}\n\n**Remediation:** {r.help_text or 'Review the artifact.'}\n\n"
            f"OWASP LLM: {', '.join(r.owasp_llm) or '-'}  \nMITRE ATLAS: "
            f"{', '.join(r.atlas) or '-'}  \nCWE: {', '.join(r.cwe) or '-'}"
        )
        rules.append(
            {
                "id": rid,
                "name": re.sub(r"[^A-Za-z0-9]+", "", r.title.title()),
                "shortDescription": {"text": r.title},
                "fullDescription": {"text": r.description},
                "help": {"text": r.description, "markdown": md},
                "helpUri": f"{REPO_URL}#rules",
                "defaultConfiguration": {"level": _LEVEL[worst[rid]]},
                "properties": {
                    "tags": tags,
                    "security-severity": _SCORE[worst[rid]],
                    "precision": "high" if rid in ("LS100", "LS150") else "medium",
                },
            }
        )
    index = {rid: i for i, rid in enumerate(rule_ids)}
    results = [_result(f, index) for f in findings]
    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "loadsafe",
                        "version": __version__,
                        "semanticVersion": __version__,
                        "informationUri": REPO_URL,
                        "rules": rules,
                    }
                },
                "results": results,
                "properties": {"summary": report.count_by_severity()},
            }
        ],
    }
    return json.dumps(sarif, indent=2)


def _result(f: Finding, index: dict[str, int]) -> dict[str, Any]:
    loc: dict[str, Any] = {"artifactLocation": {"uri": _uri(f.path)}}
    m = _LINE.match(f.location or "")
    if m:
        loc["region"] = {"startLine": int(m.group(1))}
    text = f"[{f.severity.name}] {f.message}" + (f" ({f.location})" if f.location and not m else "")
    fp = hashlib.sha256(f"{f.rule_id}|{f.path}|{f.location}|{f.message}".encode()).hexdigest()
    return {
        "ruleId": f.rule_id,
        "ruleIndex": index[f.rule_id],
        "level": _LEVEL[f.severity],
        "message": {"text": text},
        "locations": [{"physicalLocation": loc}],
        "partialFingerprints": {"loadsafe/v1": fp[:32]},
        "properties": {
            "severity": f.severity.label,
            **({"details": f.details} if f.details else {}),
        },
    }
