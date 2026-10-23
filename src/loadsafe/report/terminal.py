from __future__ import annotations

from ..findings import Finding, ScanReport, Severity
from ..rules import RULES

_COLORS = {
    Severity.CRITICAL: "\033[1;97;41m",
    Severity.HIGH: "\033[1;31m",
    Severity.MEDIUM: "\033[33m",
    Severity.LOW: "\033[36m",
    Severity.INFO: "\033[2m",
}
_RESET = "\033[0m"


def _fmt_finding(f: Finding, color: bool) -> str:
    sev = f.severity.name.ljust(8)
    if color:
        sev = f"{_COLORS[f.severity]}{sev}{_RESET}"
    rule = RULES.get(f.rule_id)
    tags = ""
    if rule and (rule.owasp_llm or rule.atlas):
        tags = " [" + ", ".join([o.split(" ")[0] for o in rule.owasp_llm] + list(rule.atlas)) + "]"
    loc = f" ({f.location})" if f.location else ""
    return f"  {sev} {f.rule_id} {f.message}{loc}{tags}"


def to_text(report: ScanReport, color: bool = False, verbose: bool = False, **_: object) -> str:
    lines = [
        f"loadsafe {report.tool_version} - scanned {len(report.artifacts)} artifact(s), "
        f"{sum(a.size for a in report.artifacts) / 1e6:.1f} MB in {report.seconds:.2f}s",
        "",
    ]
    show = Severity.INFO if verbose else Severity.LOW
    for a in report.artifacts:
        worst = a.max_severity
        status = worst.name if worst is not None and worst >= Severity.MEDIUM else "OK"
        lines.append(f"{a.path}  [{a.format}, {a.size / 1e6:.2f} MB]  {status}")
        for f in sorted(a.findings, key=lambda x: -x.severity):
            if f.severity >= show:
                lines.append(_fmt_finding(f, color))
    if report.repo_findings:
        lines.append("")
        lines.append("repository checks:")
        for f in sorted(report.repo_findings, key=lambda x: -x.severity):
            if f.severity >= show:
                lines.append(
                    _fmt_finding(f, color).replace(f"{f.rule_id} ", f"{f.rule_id} {f.path}: ", 1)
                )
    counts = report.count_by_severity()
    lines.append("")
    lines.append("summary: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return "\n".join(lines) + "\n"
