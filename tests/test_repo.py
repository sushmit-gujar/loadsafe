"""Repository / config / consumer-code checks."""

from __future__ import annotations

import json

from loadsafe.findings import Severity
from loadsafe.scanner import scan_paths


def test_trust_remote_code_and_auto_map(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Evil"],
                "auto_map": {"AutoModel": "modeling_evil.EvilModel"},
            }
        )
    )
    (tmp_path / "modeling_evil.py").write_text(
        "import os\nos.system('echo loadsafe-poc')\n"
    )
    (tmp_path / "model.safetensors").write_bytes(
        b"\x00" * 8 + b'{"__metadata__":{}}'  # intentionally broken; still a candidate
    )
    report = scan_paths([str(tmp_path)], hash_files=False)
    rules = {f.rule_id for f in report.repo_findings}
    assert "LS200" in rules
    assert "LS201" in rules


def test_chat_template_ssti(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "{{ ''.__class__.__mro__[1].__subclasses__() }}",
            }
        )
    )
    report = scan_paths([str(tmp_path)], hash_files=False)
    assert any(f.rule_id == "LS141" for f in report.repo_findings)


def test_consumer_code_unpinned(tmp_path):
    (tmp_path / "load.py").write_text(
        "from transformers import AutoModel\n"
        "m = AutoModel.from_pretrained('org/name', trust_remote_code=True)\n"
        "import torch\n"
        "torch.load('x.pt')\n"
    )
    report = scan_paths([str(tmp_path)], hash_files=False)
    rules = {f.rule_id for f in report.repo_findings}
    assert "LS211" in rules
    assert "LS210" in rules
    assert "LS212" in rules


def test_executable_flagged(tmp_path):
    (tmp_path / "payload.so").write_bytes(b"\x7fELF" + b"\x00" * 60)
    report = scan_paths([str(tmp_path)], hash_files=False)
    assert any(f.rule_id == "LS202" and f.severity >= Severity.HIGH for f in report.repo_findings)


def test_license_missing(tmp_path):
    (tmp_path / "config.json").write_text('{"architectures":["X"]}')
    (tmp_path / "model.safetensors").write_bytes(
        # minimal valid empty-header safetensors (header len 2: "{}")
        b"\x02\x00\x00\x00\x00\x00\x00\x00{}"
    )
    report = scan_paths([str(tmp_path)], hash_files=False)
    assert any(f.rule_id == "LS203" for f in report.repo_findings)
