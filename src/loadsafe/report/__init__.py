"""Report renderers."""

from __future__ import annotations

from ..findings import ScanReport
from .cyclonedx import to_cyclonedx
from .html import to_html
from .json_report import to_json
from .sarif import to_sarif
from .terminal import to_text

RENDERERS = {
    "text": to_text,
    "json": to_json,
    "sarif": to_sarif,
    "cyclonedx": to_cyclonedx,
    "html": to_html,
}


def render(report: ScanReport, fmt: str, **kw: object) -> str:
    return RENDERERS[fmt](report, **kw)  # type: ignore[operator]


__all__ = ["render", "RENDERERS"]
