"""Self-contained HTML report."""

from __future__ import annotations

import html

from ..findings import ScanReport, Severity
from ..rules import ATLAS_NAMES, RULES

_BADGE = {
    Severity.CRITICAL: "#7f1d1d",
    Severity.HIGH: "#dc2626",
    Severity.MEDIUM: "#d97706",
    Severity.LOW: "#0891b2",
    Severity.INFO: "#6b7280",
}

_CSS = """
body{font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#0b1020;
color:#e5e7eb}main{max-width:1100px;margin:0 auto;padding:32px 20px}h1{margin:0 0 4px;
font-size:26px}h2{margin-top:32px;font-size:18px;border-bottom:1px solid #273049;
padding-bottom:6px}.muted{color:#94a3b8}.cards{display:flex;gap:12px;flex-wrap:wrap;
margin:18px 0}.card{background:#131a2e;border:1px solid #273049;border-radius:10px;
padding:12px 16px;min-width:110px}.card b{display:block;font-size:22px}table{width:100%;
border-collapse:collapse;margin-top:8px}th,td{text-align:left;padding:7px 8px;
border-bottom:1px solid #1f2740;vertical-align:top}th{color:#94a3b8;font-weight:600;
font-size:12px;text-transform:uppercase;letter-spacing:.04em}.sev{display:inline-block;
padding:1px 8px;border-radius:999px;color:#fff;font-size:12px;font-weight:600}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cbd5e1;word-break:break-all}
.tag{display:inline-block;background:#1e293b;border:1px solid #334155;border-radius:6px;
padding:0 6px;margin:1px 2px 1px 0;font-size:11px;color:#cbd5e1}
"""


def _sev(s: Severity) -> str:
    return f'<span class="sev" style="background:{_BADGE[s]}">{s.name}</span>'


def to_html(report: ScanReport, title: str = "loadsafe scan report", **_: object) -> str:
    e = html.escape
    counts = report.count_by_severity()
    rows = []
    for f in sorted(report.all_findings, key=lambda x: (-x.severity, x.path)):
        r = RULES.get(f.rule_id)
        tags = ""
        if r:
            tags = "".join(f'<span class="tag">{e(t.split(" ")[0])}</span>' for t in r.owasp_llm)
            tags += "".join(
                f'<span class="tag" title="{e(ATLAS_NAMES.get(t, ""))}">{e(t)}</span>'
                for t in r.atlas
            )
            tags += "".join(f'<span class="tag">{e(c)}</span>' for c in r.cwe)
        rows.append(
            f"<tr><td>{_sev(f.severity)}</td><td><code>{e(f.rule_id)}</code><br>"
            f"<span class='muted'>{e(r.title if r else '')}</span></td><td>{e(f.message)}</td>"
            f"<td><code>{e(f.path)}</code>"
            f"{'<br><span class=muted>' + e(f.location) + '</span>' if f.location else ''}</td>"
            f"<td>{tags}</td></tr>"
        )
    arts = []
    for a in report.artifacts:
        worst = a.max_severity
        arts.append(
            f"<tr><td><code>{e(a.path)}</code></td><td>{e(a.format)}</td>"
            f"<td>{a.size / 1e6:.2f} MB</td><td>{_sev(worst) if worst is not None else 'clean'}"
            f"</td><td><code>{e(a.sha256[:16])}…</code></td></tr>"
        )
    cards = "".join(
        f'<div class="card"><span class="muted">{k}</span><b>{v}</b></div>'
        for k, v in counts.items()
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title>
<style>{_CSS}</style></head><body><main>
<h1>{e(title)}</h1>
<div class="muted">loadsafe {e(report.tool_version)} · {e(report.started_at)} ·
{len(report.artifacts)} artifacts · {sum(a.size for a in report.artifacts) / 1e6:.1f} MB ·
{report.seconds:.2f}s</div>
<div class="cards">{cards}</div>
<h2>Findings ({len(report.all_findings)})</h2>
<table><thead><tr><th>Severity</th><th>Rule</th><th>Finding</th><th>Where</th>
<th>Mapping</th></tr></thead><tbody>{"".join(rows) or "<tr><td colspan=5>No findings</td></tr>"}
</tbody></table>
<h2>Artifacts</h2>
<table><thead><tr><th>Path</th><th>Format</th><th>Size</th><th>Worst</th><th>SHA-256</th></tr>
</thead><tbody>{"".join(arts)}</tbody></table>
</main></body></html>
"""
