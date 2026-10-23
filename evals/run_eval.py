#!/usr/bin/env python3
"""Evaluate loadsafe on the inert PoC corpus + a few real public Hugging Face models.

Writes evals/results.json and prints a markdown table. Optionally compares against
picklescan / modelscan when they are installed.

Usage:
    python evals/run_eval.py                  # corpus only
    python evals/run_eval.py --hf             # + download a few small public models
    python evals/run_eval.py --hf --compare   # + compare picklescan/modelscan on corpus
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from loadsafe.findings import Severity  # noqa: E402
from loadsafe.scanner import scan_paths  # noqa: E402
from tests import corpus  # noqa: E402

# Small, popular, publicly licensed checkpoints. Mix of pickle (.bin) and safetensors.
HF_MODELS = [
    # Tiny random BERT (pickle + safetensors), ~thousands of params.
    ("hf-internal-testing/tiny-random-bert", "f171d7baecaf37b5da5a3616d8833b9969753535", "mixed"),
    # Tiny random GPT-2 (pickle + safetensors).
    ("hf-internal-testing/tiny-random-gpt2", "71034c5d8bde858ff824298bdedc65515b97d2b9", "mixed"),
    # Classic pickle-only tiny GPT-2.
    ("sshleifer/tiny-gpt2", "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be", "pickle"),
    # Popular small sentence-transformer (safetensors + pickle), ~22 MB.
    ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "mixed"),
    # Google L-2 H-128 BERT (safetensors + pickle), ~4 MB.
    ("google/bert_uncased_L-2_H-128_A-2", "30b0a37ccaaa32f332884b96992754e246e48c5f", "mixed"),
]


def _run_corpus(tmpdir: Path) -> dict:
    cases = corpus.all_cases()
    built = corpus.materialize(cases, tmpdir / "corpus")
    rows = []
    t0 = time.perf_counter()
    total_bytes = 0
    for case, path in built:
        report = scan_paths([str(path)], hash_files=False, repo_checks=False)
        worst = report.max_severity()
        flagged = worst is not None and worst >= Severity.HIGH
        total_bytes += path.stat().st_size
        rows.append(
            {
                "name": case.name,
                "malicious": case.malicious,
                "flagged": flagged,
                "max_severity": worst.label if worst else None,
                "rules": sorted({f.rule_id for f in report.all_findings}),
                "bytes": path.stat().st_size,
                "seconds": report.seconds,
            }
        )
    wall = time.perf_counter() - t0
    scan_seconds = sum(r["seconds"] for r in rows)
    tp = sum(1 for r in rows if r["malicious"] and r["flagged"])
    fn = sum(1 for r in rows if r["malicious"] and not r["flagged"])
    fp = sum(1 for r in rows if not r["malicious"] and not r["flagged"])
    tn = sum(1 for r in rows if not r["malicious"] and not r["flagged"])
    # fix fp/tn - I made a typo, restore
    fp = sum(1 for r in rows if not r["malicious"] and r["flagged"])
    tn = sum(1 for r in rows if not r["malicious"] and not r["flagged"])
    mal = tp + fn
    ben = fp + tn
    return {
        "rows": rows,
        "detection_rate": tp / mal if mal else None,
        "false_positive_rate": fp / ben if ben else None,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "seconds": scan_seconds,
        "wall_seconds": wall,
        "bytes": total_bytes,
        "mb_per_s": (total_bytes / 1e6) / scan_seconds if scan_seconds else None,
    }


def _run_hf(cache: Path) -> dict:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("pip install loadsafe[hf]") from exc

    rows = []
    t0 = time.perf_counter()
    total_bytes = 0
    for repo, rev, kind in HF_MODELS:
        print(f"  downloading {repo}" + (f"@{rev[:12]}" if rev else "") + " ...", flush=True)
        dest = cache / "local" / repo.replace("/", "__")
        local = Path(
            snapshot_download(
                repo_id=repo,
                revision=rev,
                local_dir=str(dest),
                # Materialise real files so throughput / FPR are not dominated by HF cache
                # symlink bookkeeping (LS204 INFO on snapshots/ -> blobs/).
                allow_patterns=[
                    "*.safetensors",
                    "*.bin",
                    "*.pt",
                    "*.pth",
                    "*.pkl",
                    "*.joblib",
                    "*.onnx",
                    "*.gguf",
                    "*.h5",
                    "*.json",
                    "LICENSE*",
                    "README*",
                    "*.txt",
                    "*.model",
                    "*.jinja",
                ],
            )
        )
        report = scan_paths([str(local)], hash_files=False)
        # Count only model weight files for throughput
        weight_bytes = sum(a.size for a in report.artifacts)
        total_bytes += weight_bytes
        high = [f for f in report.all_findings if f.severity >= Severity.HIGH]
        # For FPR on public models: HIGH findings that are NOT the expected "pickle format"
        # advisories on known-pickle repos. LS107/LS213 are LOW. Anything HIGH on a
        # well-known public model is treated as a false positive for this eval.
        rows.append(
            {
                "repo": repo,
                "revision": rev,
                "kind": kind,
                "artifacts": len(report.artifacts),
                "bytes": weight_bytes,
                "seconds": report.seconds,
                "findings": report.count_by_severity(),
                "high_findings": [
                    {"rule": f.rule_id, "msg": f.message, "path": f.path} for f in high
                ],
                "false_positive": bool(high),
            }
        )
    wall = time.perf_counter() - t0
    scan_seconds = sum(r["seconds"] for r in rows)
    return {
        "rows": rows,
        "false_positive_rate": (
            sum(1 for r in rows if r["false_positive"]) / len(rows) if rows else None
        ),
        "seconds": scan_seconds,
        "wall_seconds": wall,
        "bytes": total_bytes,
        "mb_per_s": (total_bytes / 1e6) / scan_seconds if scan_seconds else None,
    }


def _compare(tmpdir: Path) -> dict:
    cases = corpus.malicious_cases() + corpus.benign_cases()
    built = corpus.materialize(cases, tmpdir / "compare")
    out: dict = {}
    for tool in ("picklescan", "modelscan"):
        if shutil.which(tool) is None and tool == "picklescan":
            # picklescan installs a console script
            try:
                import picklescan  # noqa: F401
            except ImportError:
                out[tool] = {"available": False}
                continue
        if tool == "modelscan" and shutil.which("modelscan") is None:
            out[tool] = {"available": False}
            continue
        tp = fp = fn = tn = 0
        t0 = time.perf_counter()
        for case, path in built:
            flagged = _tool_flags(tool, path)
            if case.malicious and flagged:
                tp += 1
            elif case.malicious and not flagged:
                fn += 1
            elif not case.malicious and flagged:
                fp += 1
            else:
                tn += 1
        seconds = time.perf_counter() - t0
        mal, ben = tp + fn, fp + tn
        out[tool] = {
            "available": True,
            "detection_rate": tp / mal if mal else None,
            "false_positive_rate": fp / ben if ben else None,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "seconds": seconds,
        }
    return out


def _tool_flags(tool: str, path: Path) -> bool:
    if tool == "picklescan":
        r = subprocess.run(
            ["picklescan", "-p", str(path)],
            capture_output=True,
            text=True,
        )
        text = r.stdout + r.stderr
        # picklescan exits non-zero on parse errors too; count only real hits.
        if "Infected files: 0" in text and "Dangerous globals: 0" in text:
            return False
        return ("Infected files:" in text and "Infected files: 0" not in text) or (
            "Dangerous globals:" in text and "Dangerous globals: 0" not in text
        )
    if tool == "modelscan":
        r = subprocess.run(
            ["modelscan", "-p", str(path), "-r", "json"],
            capture_output=True,
            text=True,
        )
        # modelscan prints a banner before the JSON document.
        raw = r.stdout
        brace = raw.find("{")
        try:
            data = json.loads(raw[brace:]) if brace >= 0 else {}
        except json.JSONDecodeError:
            return "CRITICAL" in raw or "Total Issues: " in raw and "Total Issues: 0" not in raw
        by_sev = (data.get("summary") or {}).get("total_issues_by_severity") or {}
        return any(by_sev.get(k, 0) > 0 for k in ("CRITICAL", "HIGH", "MEDIUM"))
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true", help="scan real public HF models")
    ap.add_argument("--compare", action="store_true", help="compare picklescan/modelscan")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "evals" / "results.json")
    args = ap.parse_args()

    results: dict = {
        "tool": "loadsafe",
        "version": __import__("loadsafe").__version__,
        "setup": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "note": (
                "PoC malicious artifacts are generated at eval time; payloads are inert "
                "(print / write a marker file). Real HF models are downloaded read-only."
            ),
        },
    }
    with tempfile.TemporaryDirectory(prefix="loadsafe-eval-") as td:
        tmp = Path(td)
        print("== corpus ==")
        results["corpus"] = _run_corpus(tmp)
        c = results["corpus"]
        print(
            f"  detection={c['detection_rate']:.0%}  FPR={c['false_positive_rate']:.0%}  "
            f"{c['mb_per_s']:.1f} MB/s  ({c['tp']}/{c['tp']+c['fn']} mal, "
            f"{c['fp']}/{c['fp']+c['tn']} fp/benign)"
        )
        if args.compare:
            print("== compare ==")
            results["compare"] = _compare(tmp)
            for name, row in results["compare"].items():
                if not row.get("available"):
                    print(f"  {name}: not installed")
                    continue
                print(
                    f"  {name}: detection={row['detection_rate']:.0%}  "
                    f"FPR={row['false_positive_rate']:.0%}  ({row['seconds']:.2f}s)"
                )
        if args.hf:
            print("== huggingface ==")
            cache = ROOT / "evals" / ".cache"
            cache.mkdir(parents=True, exist_ok=True)
            results["huggingface"] = _run_hf(cache)
            h = results["huggingface"]
            print(
                f"  FPR={h['false_positive_rate']:.0%}  {h['mb_per_s']:.1f} MB/s  "
                f"({h['bytes']/1e6:.1f} MB over {len(h['rows'])} repos)"
            )
            for r in h["rows"]:
                flag = "FP" if r["false_positive"] else "ok"
                print(
                    f"    [{flag}] {r['repo']}: {r['artifacts']} artifacts, "
                    f"{r['bytes']/1e6:.1f} MB, {r['seconds']:.2f}s, findings={r['findings']}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
