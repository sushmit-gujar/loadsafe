"""Command-line interface: ``loadsafe scan PATH... [--format ...] [--fail-on high]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .findings import Severity
from .policy import Policy
from .report import RENDERERS, render
from .rules import RULES
from .scanner import scan_paths

SEVERITIES = [s.label for s in Severity]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loadsafe",
        description="Audit ML model artifacts and model repositories before loading them.",
    )
    p.add_argument("--version", action="version", version=f"loadsafe {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan files, directories or hf://org/name[@rev] repos")
    s.add_argument("targets", nargs="+")
    s.add_argument("-f", "--format", choices=sorted(RENDERERS), default="text")
    s.add_argument("-o", "--output", help="write the report to this file (default: stdout)")
    s.add_argument("--sarif", help="additionally write SARIF to this file")
    s.add_argument(
        "--fail-on",
        choices=SEVERITIES + ["never"],
        default="high",
        help="exit 1 if any finding is at or above this severity (default: high)",
    )
    s.add_argument(
        "--min-severity",
        choices=SEVERITIES,
        default="info",
        help="drop findings below this severity from the report",
    )
    s.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="MODULE.NAME",
        help="extra allowlisted pickle global ('pkg.mod.Class' or 'pkg.*')",
    )
    s.add_argument("--deny", action="append", default=[], metavar="MODULE.NAME")
    s.add_argument("--revision", help="default revision for hf:// targets")
    s.add_argument("--no-repo-checks", action="store_true")
    s.add_argument("--no-hash", action="store_true", help="skip SHA-256 hashing")
    s.add_argument("-v", "--verbose", action="store_true", help="show INFO findings in text")
    sub.add_parser("rules", help="list rules and their OWASP / MITRE ATLAS mappings")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "rules":
        for r in RULES.values():
            print(f"{r.id}  {r.title}")
            print(
                f"       OWASP: {', '.join(r.owasp_llm) or '-'} | ATLAS: "
                f"{', '.join(r.atlas) or '-'} | CWE: {', '.join(r.cwe) or '-'}"
            )
        return 0
    policy = Policy(extra_allow=set(args.allow), extra_deny=set(args.deny))
    try:
        report = scan_paths(
            args.targets,
            policy,
            revision=args.revision,
            repo_checks=not args.no_repo_checks,
            hash_files=not args.no_hash,
            min_severity=Severity.parse(args.min_severity),
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"loadsafe: error: {exc}", file=sys.stderr)
        return 2
    color = args.output is None and sys.stdout.isatty()
    out = render(report, args.format, color=color, verbose=args.verbose)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        if args.format != "text":
            print(render(report, "text", color=sys.stdout.isatty()), end="")
    else:
        sys.stdout.write(out)
    if args.sarif:
        Path(args.sarif).write_text(render(report, "sarif"), encoding="utf-8")
    if args.fail_on != "never" and report.fails(Severity.parse(args.fail_on)):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
