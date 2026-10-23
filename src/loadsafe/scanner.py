"""Top-level orchestration: files, directories and hf:// repositories."""

from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path

from . import __version__
from .findings import ArtifactResult, Finding, ScanReport, Severity
from .policy import Policy
from .repo_scan import CONFIG_FILES, scan_repo
from .scanners.artifact import (
    MODEL_EXTENSIONS,
    NON_PICKLE_PROMISES,
    PICKLE_BASED,
    is_candidate,
    iter_files,
    scan_file,
)

WEIGHT_HINTS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
    ".onnx",
    ".h5",
    ".keras",
    ".ckpt",
    ".pkl",
    ".joblib",
    ".npz",
    ".npy",
)


def resolve_target(target: str, revision: str | None = None) -> tuple[Path, str]:
    """Return a local path for ``target``; ``hf://org/name[@rev]`` is downloaded first."""
    if target.startswith("hf://"):
        spec = target[5:]
        repo, _, rev = spec.partition("@")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("hf:// targets need `pip install loadsafe[hf]`") from exc
        local = snapshot_download(repo_id=repo, revision=rev or revision)
        return Path(local), target
    return Path(target), target


def scan_paths(
    targets: list[str],
    policy: Policy | None = None,
    revision: str | None = None,
    repo_checks: bool = True,
    hash_files: bool = True,
    min_severity: Severity = Severity.INFO,
) -> ScanReport:
    policy = policy or Policy()
    t0 = time.perf_counter()
    report = ScanReport(
        targets=list(targets),
        tool_version=__version__,
        started_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    for target in targets:
        path, label = resolve_target(target, revision)
        if path.is_file():
            res = scan_file(path, str(label), policy, hash_files)
            report.artifacts.append(res)
            continue
        if not path.is_dir():
            report.repo_findings.append(
                Finding("LS299", Severity.MEDIUM, "target not found", str(label))
            )
            continue
        files = iter_files(path)
        prefix = label.rstrip("/")
        artifacts: list[ArtifactResult] = []
        for p in files:
            if p.name in CONFIG_FILES:
                continue
            # Hugging Face hub caches use symlinks from snapshots/ into blobs/. Follow the
            # target for content scanning; repo_scan reports LS204 for escapes.
            try:
                target = p.resolve() if p.is_symlink() else p
            except OSError:
                continue
            if not target.is_file():
                continue
            if not is_candidate(target):
                continue
            rel = p.relative_to(path).as_posix()
            res = scan_file(target, f"{prefix}/{rel}", policy, hash_files)
            if res.format == "unknown" and p.suffix.lower() not in NON_PICKLE_PROMISES:
                continue
            artifacts.append(res)
        report.artifacts.extend(artifacts)
        if repo_checks:
            looks_like_model = (path / "config.json").is_file() or any(
                p.suffix.lower() in WEIGHT_HINTS for p in files
            )
            repo = scan_repo(path, files, looks_like_model)
            for f in repo:
                f.path = f"{prefix}/{f.path}" if f.path != "." else prefix
            pickled = [a for a in artifacts if a.format in PICKLE_BASED]
            if pickled and not any(a.format == "safetensors" for a in artifacts):
                repo.append(
                    Finding(
                        "LS213",
                        Severity.LOW,
                        f"{len(pickled)} pickle-based weight "
                        "file(s) and no safetensors alternative",
                        prefix,
                    )
                )
            report.repo_findings.extend(repo)
    if min_severity > Severity.INFO:
        for a in report.artifacts:
            a.findings = [f for f in a.findings if f.severity >= min_severity]
        report.repo_findings = [f for f in report.repo_findings if f.severity >= min_severity]
    report.seconds = time.perf_counter() - t0
    return report


__all__ = ["scan_paths", "resolve_target", "MODEL_EXTENSIONS"]
