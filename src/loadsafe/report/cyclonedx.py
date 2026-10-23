"""CycloneDX 1.6 ML-BOM: model components with hashes plus findings as vulnerabilities."""

from __future__ import annotations

import json
import uuid
from typing import Any

from .. import __version__
from ..findings import ScanReport
from ..rules import RULES


def _cwe_num(c: str) -> int | None:
    try:
        return int(c.split("-", 1)[1])
    except (IndexError, ValueError):
        return None


def to_cyclonedx(report: ScanReport, **_: object) -> str:
    components: list[dict[str, Any]] = []
    refs: dict[str, str] = {}
    for i, a in enumerate(report.artifacts):
        ref = f"artifact-{i}"
        refs[a.path] = ref
        props = [
            {"name": "loadsafe:path", "value": a.path},
            {"name": "loadsafe:format", "value": a.format},
            {"name": "loadsafe:size_bytes", "value": str(a.size)},
            {
                "name": "loadsafe:max_severity",
                "value": a.max_severity.label if a.max_severity is not None else "none",
            },
        ]
        for key in (
            "parameters",
            "tensors",
            "gguf_version",
            "general.architecture",
            "producer_name",
            "ir_version",
        ):
            if key in a.metadata:
                props.append({"name": f"loadsafe:{key}", "value": str(a.metadata[key])})
        comp: dict[str, Any] = {
            "type": "machine-learning-model",
            "bom-ref": ref,
            "name": a.path.rsplit("/", 1)[-1],
            "properties": props,
        }
        if a.sha256:
            comp["hashes"] = [{"alg": "SHA-256", "content": a.sha256}]
        arch = a.metadata.get("general.architecture")
        if arch:
            comp["modelCard"] = {"modelParameters": {"modelArchitecture": str(arch)}}
        components.append(comp)
    vulns: list[dict[str, Any]] = []
    for n, f in enumerate(report.all_findings):
        r = RULES.get(f.rule_id)
        target = refs.get(f.path)
        if target is None:  # repo-level finding: attach to all artifacts under that path
            affected = [ref for p, ref in refs.items() if p.startswith(f.path.rstrip("/") + "/")]
        else:
            affected = [target]
        v: dict[str, Any] = {
            "bom-ref": f"finding-{n}",
            "id": f.rule_id,
            "source": {"name": "loadsafe", "url": "https://github.com/sushmit-gujar/loadsafe"},
            "ratings": [
                {"severity": f.severity.label, "method": "other", "source": {"name": "loadsafe"}}
            ],
            "description": f.message,
            "detail": f"{f.path}" + (f" ({f.location})" if f.location else ""),
            "properties": [{"name": "loadsafe:path", "value": f.path}],
        }
        if r is not None:
            cwes = [c for c in (_cwe_num(x) for x in r.cwe) if c is not None]
            if cwes:
                v["cwes"] = cwes
            if r.help_text:
                v["recommendation"] = r.help_text
            v["properties"] += [{"name": "owasp:llm", "value": o} for o in r.owasp_llm] + [
                {"name": "mitre:atlas", "value": t} for t in r.atlas
            ]
        if affected:
            v["affects"] = [{"ref": ref} for ref in affected]
        vulns.append(v)
    metadata: dict[str, Any] = {
        "tools": {
            "components": [{"type": "application", "name": "loadsafe", "version": __version__}]
        },
    }
    if report.started_at:
        metadata["timestamp"] = report.started_at
    bom: dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": metadata,
        "components": components,
        "vulnerabilities": vulns,
    }
    return json.dumps(bom, indent=2)
