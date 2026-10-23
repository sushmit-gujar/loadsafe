"""End-to-end: every malicious PoC is flagged HIGH+, every benign artifact is clean of HIGH+."""

from __future__ import annotations

import pytest

from loadsafe.findings import Severity
from loadsafe.scanner import scan_paths

from . import corpus

pytest.importorskip("numpy")


def _skip_if_missing(case: corpus.Case) -> None:
    needs = {
        "sklearn": ("sklearn",), "torch": ("torch",), "safetensors": ("safetensors",),
        "gguf": ("gguf",), "onnx": ("onnx",),
    }
    for mod, markers in needs.items():
        if any(m in case.name for m in markers):
            pytest.importorskip(mod)


@pytest.mark.parametrize("case", corpus.malicious_cases(), ids=lambda c: c.name)
def test_malicious_detected(case: corpus.Case, tmp_path):
    _skip_if_missing(case)
    path = case.build(tmp_path)
    report = scan_paths([str(path)])
    worst = report.max_severity()
    assert worst is not None and worst >= Severity.HIGH, (
        f"{case.name}: expected HIGH+, got {worst}")
    rule_ids = {f.rule_id for f in report.all_findings}
    if case.expect_rules:
        assert rule_ids & set(case.expect_rules), (
            f"{case.name}: expected one of {case.expect_rules}, got {sorted(rule_ids)}")


@pytest.mark.parametrize("case", corpus.benign_cases(), ids=lambda c: c.name)
def test_benign_clean(case: corpus.Case, tmp_path):
    _skip_if_missing(case)
    path = case.build(tmp_path)
    report = scan_paths([str(path)])
    high = [f for f in report.all_findings if f.severity >= Severity.HIGH]
    assert not high, f"{case.name}: false positive(s): {[(f.rule_id, f.message) for f in high]}"


def test_scan_directory_mixed(tmp_path):
    corpus.materialize(corpus.benign_cases()[:3] + corpus.malicious_cases()[:3], tmp_path)
    report = scan_paths([str(tmp_path)], hash_files=True)
    assert report.fails(Severity.HIGH)
    assert len(report.artifacts) >= 6
    assert all(a.sha256 for a in report.artifacts)
