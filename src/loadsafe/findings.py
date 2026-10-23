"""Core data model: severities, rules and findings."""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


class Severity(enum.IntEnum):
    """Ordered severity levels (higher is worse)."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, value: str) -> Severity:
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:  # pragma: no cover - argparse guards this
            raise ValueError(f"unknown severity {value!r}") from exc

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Rule:
    """A detection rule with its threat-framework mapping."""

    id: str
    title: str
    description: str
    owasp_llm: tuple[str, ...] = ()
    atlas: tuple[str, ...] = ()
    cwe: tuple[str, ...] = ()
    help_text: str = ""


@dataclass
class Finding:
    """One issue found in one artifact."""

    rule_id: str
    severity: Severity
    message: str
    path: str
    location: str | None = None  # e.g. "archive/data.pkl@offset 123" or a JSON key
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.label
        return d


@dataclass
class ArtifactResult:
    """Scan result for one file."""

    path: str
    format: str
    size: int
    sha256: str
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    scan_seconds: float = 0.0
    error: str | None = None

    @property
    def max_severity(self) -> Severity | None:
        return max((f.severity for f in self.findings), default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "size": self.size,
            "sha256": self.sha256,
            "scan_seconds": round(self.scan_seconds, 6),
            "error": self.error,
            "metadata": self.metadata,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class ScanReport:
    """Aggregate result over all scanned targets."""

    targets: list[str]
    artifacts: list[ArtifactResult] = field(default_factory=list)
    repo_findings: list[Finding] = field(default_factory=list)
    tool_version: str = ""
    started_at: str = ""
    seconds: float = 0.0

    @property
    def all_findings(self) -> list[Finding]:
        out: list[Finding] = list(self.repo_findings)
        for a in self.artifacts:
            out.extend(a.findings)
        return out

    def count_by_severity(self) -> dict[str, int]:
        counts = {s.label: 0 for s in sorted(Severity, reverse=True)}
        for f in self.all_findings:
            counts[f.severity.label] += 1
        return counts

    def max_severity(self) -> Severity | None:
        return max((f.severity for f in self.all_findings), default=None)

    def fails(self, threshold: Severity) -> bool:
        worst = self.max_severity()
        return worst is not None and worst >= threshold

    def to_dict(self) -> dict[str, Any]:
        total_bytes = sum(a.size for a in self.artifacts)
        return {
            "tool": {"name": "loadsafe", "version": self.tool_version},
            "started_at": self.started_at,
            "seconds": round(self.seconds, 6),
            "targets": self.targets,
            "summary": {
                "artifacts": len(self.artifacts),
                "bytes": total_bytes,
                "findings": self.count_by_severity(),
                "max_severity": (self.max_severity().label if self.max_severity() is not None
                                 else None),
            },
            "repo_findings": [f.to_dict() for f in self.repo_findings],
            "artifacts": [a.to_dict() for a in self.artifacts],
        }
