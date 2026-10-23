"""ONNX checks via a minimal protobuf wire-format walker (no onnx package required)."""

from __future__ import annotations

import mmap
import posixpath
from collections.abc import Iterator

from ..findings import Severity
from .common import Ctx

STANDARD_DOMAINS = frozenset(
    {
        "",
        "ai.onnx",
        "ai.onnx.ml",
        "ai.onnx.training",
        "ai.onnx.preview.training",
        "com.microsoft",
        "com.microsoft.nchwc",
        "com.microsoft.experimental",
        "com.ms.internal.nhwc",
        "org.pytorch.aten",
        "com.microsoft.dml",
        "ai.onnx.contrib",
    }
)
PYTHON_OPS = frozenset({"PythonOp", "PythonOpGrad"})
MAX_DEPTH = 24


class _Bad(Exception):
    pass


def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise _Bad("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise _Bad("varint too long")


def _fields(buf: memoryview) -> Iterator[tuple[int, int, int | memoryview]]:
    pos = 0
    n = len(buf)
    while pos < n:
        key, pos = _varint(buf, pos)
        field, wt = key >> 3, key & 7
        if field == 0:
            raise _Bad("field number 0")
        if wt == 0:
            val, pos = _varint(buf, pos)
            yield field, wt, val
        elif wt == 1:
            if pos + 8 > n:
                raise _Bad("truncated fixed64")
            pos += 8
            yield field, wt, 0
        elif wt == 2:
            ln, pos = _varint(buf, pos)
            if pos + ln > n:
                raise _Bad(f"length-delimited field {field} overruns buffer")
            yield field, wt, buf[pos : pos + ln]
            pos += ln
        elif wt == 5:
            if pos + 4 > n:
                raise _Bad("truncated fixed32")
            pos += 4
            yield field, wt, 0
        else:
            raise _Bad(f"unsupported wire type {wt}")


def _s(v: int | memoryview) -> str:
    return bytes(v).decode("utf-8", errors="replace") if isinstance(v, memoryview) else str(v)


class _Walker:
    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.local_functions: set[str] = set()
        self.domains: dict[str, int] = {}
        self.ops = 0
        self.external = 0
        self.initializers = 0

    def model(self, buf: memoryview) -> None:
        graphs: list[memoryview] = []
        functions: list[memoryview] = []
        meta: dict[str, object] = {}
        for field, wt, v in _fields(buf):
            if field == 1 and wt == 0:
                meta["ir_version"] = v
            elif field == 2 and wt == 2:
                meta["producer_name"] = _s(v)[:100]
            elif field == 7 and wt == 2:
                graphs.append(v)  # type: ignore[arg-type]
            elif field == 8 and wt == 2:
                dom, ver = "", 0
                for f2, _w2, v2 in _fields(v):  # type: ignore[arg-type]
                    if f2 == 1:
                        dom = _s(v2)
                    elif f2 == 2 and isinstance(v2, int):
                        ver = v2
                meta.setdefault("opsets", {})[dom or "ai.onnx"] = ver  # type: ignore[index]
            elif field == 25 and wt == 2:
                functions.append(v)  # type: ignore[arg-type]
        if "ir_version" not in meta or not graphs:
            raise _Bad("no ir_version/graph: not an ONNX ModelProto")
        for fn in functions:
            name = dom = ""
            for f2, _w2, v2 in _fields(fn):
                if f2 == 1:
                    name = _s(v2)
                elif f2 == 10:
                    dom = _s(v2)
            self.local_functions.add(f"{dom}::{name}")
        for fn in functions:
            for f2, w2, v2 in _fields(fn):
                if f2 == 7 and w2 == 2:
                    self.node(v2, 1)  # type: ignore[arg-type]
        for g in graphs:
            self.graph(g, 0)
        self.ctx.metadata.update(meta)
        self.ctx.metadata.update(
            nodes=self.ops,
            initializers=self.initializers,
            external_tensors=self.external,
            domains=self.domains,
        )

    def graph(self, buf: memoryview, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise _Bad("graph nesting too deep")
        for field, wt, v in _fields(buf):
            if wt != 2:
                continue
            if field == 1:
                self.node(v, depth)  # type: ignore[arg-type]
            elif field == 5:
                self.initializers += 1
                self.tensor(v)  # type: ignore[arg-type]
            elif field == 15:
                for f2, w2, v2 in _fields(v):  # type: ignore[arg-type]
                    if f2 in (1, 2) and w2 == 2:
                        self.tensor(v2)  # type: ignore[arg-type]

    def node(self, buf: memoryview, depth: int) -> None:
        self.ops += 1
        op_type = name = domain = ""
        attrs: list[memoryview] = []
        for field, wt, v in _fields(buf):
            if field == 3 and wt == 2:
                name = _s(v)
            elif field == 4 and wt == 2:
                op_type = _s(v)
            elif field == 7 and wt == 2:
                domain = _s(v)
            elif field == 5 and wt == 2:
                attrs.append(v)  # type: ignore[arg-type]
        self.domains[domain or "ai.onnx"] = self.domains.get(domain or "ai.onnx", 0) + 1
        loc = f"node {name or op_type}"
        if op_type in PYTHON_OPS:
            self.ctx.add(
                "LS131",
                Severity.HIGH,
                f"{op_type} operator executes Python code at runtime",
                loc,
                domain=domain,
            )
        elif domain not in STANDARD_DOMAINS and f"{domain}::{op_type}" not in self.local_functions:
            self.ctx.add(
                "LS131",
                Severity.MEDIUM,
                f"operator {domain}::{op_type} needs a custom-op library",
                loc,
                domain=domain,
                op_type=op_type,
            )
        for a in attrs:
            for f2, w2, v2 in _fields(a):
                if w2 != 2:
                    continue
                if f2 == 5 or f2 == 10:
                    self.tensor(v2)  # type: ignore[arg-type]
                elif f2 == 6 or f2 == 11:
                    self.graph(v2, depth + 1)  # type: ignore[arg-type]

    def tensor(self, buf: memoryview) -> None:
        name = ""
        ext: dict[str, str] = {}
        for field, wt, v in _fields(buf):
            if field == 8 and wt == 2:
                name = _s(v)
            elif field == 13 and wt == 2:
                k = val = ""
                for f2, _w2, v2 in _fields(v):  # type: ignore[arg-type]
                    if f2 == 1:
                        k = _s(v2)
                    elif f2 == 2:
                        val = _s(v2)
                ext[k] = val
        if not ext:
            return
        self.external += 1
        loc_ = ext.get("location", "")
        norm = posixpath.normpath(loc_.replace("\\", "/"))
        if (
            loc_.startswith(("/", "\\"))
            or (len(loc_) > 1 and loc_[1] == ":")
            or norm == ".."
            or norm.startswith("../")
        ):
            self.ctx.add(
                "LS130",
                Severity.HIGH,
                f"external data location {loc_!r} escapes the model directory",
                f"tensor {name}",
                external_location=loc_,
            )


def scan_onnx(data: bytes | memoryview | mmap.mmap, ctx: Ctx) -> None:
    try:
        _Walker(ctx).model(memoryview(data))
    except _Bad as exc:
        ctx.add("LS132", Severity.MEDIUM, f"could not parse ONNX protobuf: {exc}")
