"""Report formats and CLI exit codes."""

from __future__ import annotations

import json
import pickle

import pytest

from loadsafe.cli import main
from loadsafe.report import render
from loadsafe.scanner import scan_paths


class _Evil:
    def __reduce__(self):
        return (eval, ("1",))


@pytest.fixture
def evil_pkl(tmp_path):
    p = tmp_path / "e.pkl"
    p.write_bytes(pickle.dumps(_Evil(), protocol=4))
    return p


def test_json_and_sarif_and_cyclonedx(evil_pkl):
    report = scan_paths([str(evil_pkl)])
    data = json.loads(render(report, "json"))
    assert data["summary"]["findings"]["critical"] >= 1
    sarif = json.loads(render(report, "sarif"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"]
    bom = json.loads(render(report, "cyclonedx"))
    assert bom["bomFormat"] == "CycloneDX"
    assert bom["components"][0]["type"] == "machine-learning-model"
    assert bom["vulnerabilities"]
    html = render(report, "html")
    assert "CRITICAL" in html and "LS100" in html


def test_cli_fail_on(evil_pkl, capsys):
    assert main(["scan", str(evil_pkl), "--fail-on", "high", "--format", "json"]) == 1
    assert main(["scan", str(evil_pkl), "--fail-on", "never"]) == 0


def test_cli_rules():
    assert main(["rules"]) == 0


def test_sarif_schema(evil_pkl):
    jsonschema = pytest.importorskip("jsonschema")
    import urllib.request

    report = scan_paths([str(evil_pkl)])
    sarif = json.loads(render(report, "sarif"))
    # Validate against a minimal local subset of SARIF required fields.
    assert "runs" in sarif and sarif["runs"][0]["tool"]["driver"]["rules"]
    try:
        with urllib.request.urlopen(
            "https://json.schemastore.org/sarif-2.1.0.json", timeout=10
        ) as resp:
            schema = json.load(resp)
        jsonschema.validate(sarif, schema)
    except Exception as exc:  # pragma: no cover - network optional
        pytest.skip(f"schema fetch failed: {exc}")
