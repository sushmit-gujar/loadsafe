from __future__ import annotations

import json

from ..findings import ScanReport
from ..rules import RULES


def to_json(report: ScanReport, **_: object) -> str:
    data = report.to_dict()
    used = sorted({f.rule_id for f in report.all_findings})
    data["rules"] = {
        rid: {
            "title": RULES[rid].title,
            "owasp_llm": list(RULES[rid].owasp_llm),
            "atlas": list(RULES[rid].atlas),
            "cwe": list(RULES[rid].cwe),
        }
        for rid in used
        if rid in RULES
    }
    return json.dumps(data, indent=2, default=str)
