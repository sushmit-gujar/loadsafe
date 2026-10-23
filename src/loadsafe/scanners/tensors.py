"""safetensors and NumPy .npy header validation."""

from __future__ import annotations

import ast
import io
import json
import struct
from typing import IO, Any

from ..findings import Severity
from ..pickle_scan import scan_pickle
from .common import CODE_LIKE, LONG_B64, URL_RE, Ctx

SAFETENSORS_DTYPES: dict[str, float] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
    "F4": 0.5,
    "F6_E2M3": 0.75,
    "F6_E3M2": 0.75,
}
MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024  # same limit as the reference implementation


def _dupe_check(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f"duplicate key {k!r}")
        seen[k] = v
    return seen


def scan_safetensors(f: IO[bytes], size: int, ctx: Ctx) -> None:
    head = f.read(8)
    if len(head) < 8:
        ctx.add("LS120", Severity.HIGH, "file shorter than the 8-byte safetensors header")
        return
    (n,) = struct.unpack("<Q", head)
    ctx.metadata["header_bytes"] = n
    if n > MAX_SAFETENSORS_HEADER:
        ctx.add("LS120", Severity.HIGH, f"header length {n} exceeds 100 MiB limit")
        return
    if n > size - 8:
        ctx.add("LS120", Severity.HIGH, f"header length {n} exceeds file size {size}")
        return
    raw = f.read(n)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        ctx.add("LS120", Severity.HIGH, "header is not valid UTF-8")
        return
    if not text.lstrip().startswith("{"):
        ctx.add("LS120", Severity.HIGH, "header does not start with '{'")
        return
    try:
        header = json.loads(text, object_pairs_hook=_dupe_check)
    except ValueError as exc:
        ctx.add("LS120", Severity.HIGH, f"header is not valid JSON: {exc}")
        return
    if not isinstance(header, dict):
        ctx.add("LS120", Severity.HIGH, "header is not a JSON object")
        return
    data_len = size - 8 - n
    meta = header.pop("__metadata__", None)
    if meta is not None:
        _check_metadata(meta, ctx)
    spans: list[tuple[int, int, str]] = []
    n_params = 0
    for name, info in header.items():
        loc = f"tensor {name}"
        if not isinstance(info, dict):
            ctx.add("LS120", Severity.HIGH, "tensor entry is not an object", loc)
            continue
        dtype, shape, offs = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if dtype not in SAFETENSORS_DTYPES:
            ctx.add("LS120", Severity.MEDIUM, f"unknown dtype {dtype!r}", loc)
            continue
        if not (
            isinstance(shape, list)
            and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)
        ):
            ctx.add("LS120", Severity.HIGH, f"invalid shape {shape!r}", loc)
            continue
        if not (
            isinstance(offs, list)
            and len(offs) == 2
            and all(isinstance(o, int) and not isinstance(o, bool) and o >= 0 for o in offs)
        ):
            ctx.add("LS120", Severity.HIGH, f"invalid data_offsets {offs!r}", loc)
            continue
        begin, end = offs
        count = 1
        for d in shape:
            count *= d
        n_params += count
        expected = count * SAFETENSORS_DTYPES[dtype]
        if begin > end or end > data_len:
            ctx.add(
                "LS120",
                Severity.HIGH,
                f"data_offsets {offs} outside data buffer of {data_len} bytes",
                loc,
            )
            continue
        if expected != int(expected) or end - begin != int(expected):
            ctx.add(
                "LS120",
                Severity.HIGH,
                f"byte span {end - begin} does not match "
                f"shape {shape} x {dtype} ({expected:g} bytes)",
                loc,
            )
        spans.append((begin, end, name))
    spans.sort()
    cursor = 0
    for begin, end, name in spans:
        if begin < cursor:
            ctx.add(
                "LS120", Severity.HIGH, "tensor data overlaps the previous tensor", f"tensor {name}"
            )
        elif begin > cursor:
            ctx.add(
                "LS120",
                Severity.MEDIUM,
                f"{begin - cursor}-byte gap before tensor data (spec forbids holes)",
                f"tensor {name}",
            )
        cursor = max(cursor, end)
    if cursor < data_len:
        ctx.add(
            "LS120",
            Severity.MEDIUM,
            f"{data_len - cursor} unreferenced bytes after the last tensor (hidden data)",
        )
    ctx.metadata.update(tensors=len(header), parameters=n_params)


def _check_metadata(meta: Any, ctx: Ctx) -> None:
    if not isinstance(meta, dict):
        ctx.add("LS121", Severity.MEDIUM, "__metadata__ is not an object")
        return
    for k, v in meta.items():
        loc = f"__metadata__.{k}"
        if not isinstance(v, str):
            ctx.add(
                "LS121", Severity.MEDIUM, "metadata value is not a string (spec violation)", loc
            )
            continue
        if CODE_LIKE.search(v):
            ctx.add(
                "LS121",
                Severity.MEDIUM,
                "metadata value contains code-like text",
                loc,
                excerpt=v[:120],
            )
        elif LONG_B64.search(v):
            ctx.add("LS121", Severity.LOW, "metadata value contains a long encoded blob", loc)
        elif len(v) > 65536:
            ctx.add("LS121", Severity.LOW, f"metadata value is {len(v)} characters long", loc)
        elif (url := URL_RE.search(v)) is not None:
            ctx.add(
                "LS121", Severity.INFO, "metadata value contains a URL", loc, url=url.group(0)[:200]
            )
    ctx.metadata["safetensors_metadata_keys"] = sorted(meta)[:50]


NPY_MAGIC = b"\x93NUMPY"


def scan_npy(f: IO[bytes], size: int, ctx: Ctx, location: str = "") -> None:
    start = f.tell()
    magic = f.read(6)
    ver = f.read(2)
    loc = location or None
    if magic != NPY_MAGIC or len(ver) != 2:
        ctx.add("LS120", Severity.HIGH, "bad .npy magic", loc)
        return
    major = ver[0]
    if major == 1:
        (hlen,) = struct.unpack("<H", f.read(2))
    elif major in (2, 3):
        (hlen,) = struct.unpack("<I", f.read(4))
    else:
        ctx.add("LS120", Severity.HIGH, f"unsupported .npy version {major}", loc)
        return
    if hlen > min(size, 1 << 20):
        ctx.add("LS120", Severity.HIGH, f".npy header length {hlen} is implausible", loc)
        return
    raw = f.read(hlen)
    try:
        header = ast.literal_eval(raw.decode("latin-1"))
    except (ValueError, SyntaxError) as exc:
        ctx.add("LS120", Severity.HIGH, f".npy header is not a Python literal: {exc}", loc)
        return
    if not isinstance(header, dict) or not {"descr", "fortran_order", "shape"} <= set(header):
        ctx.add("LS120", Severity.HIGH, ".npy header misses descr/fortran_order/shape", loc)
        return
    descr = header["descr"]
    shape = header["shape"]
    if not (isinstance(shape, tuple) and all(isinstance(d, int) and d >= 0 for d in shape)):
        ctx.add("LS120", Severity.HIGH, f"invalid .npy shape {shape!r}", loc)
        return
    data_start = f.tell()
    remaining = size - (data_start - start)
    if _descr_has_object(descr):
        ctx.metadata["npy_object_array"] = True
        ctx.add(
            "LS107",
            Severity.MEDIUM,
            "object-dtype array: np.load needs allow_pickle=True and unpickles the payload",
            loc,
        )
        res = scan_pickle(
            f, ctx.path, f"{location}/pickle" if location else "npy-data", ctx.policy, multi=False
        )
        ctx.findings.extend(res.findings)
        return
    itemsize = _descr_itemsize(descr)
    if itemsize is not None:
        count = 1
        for d in shape:
            count *= d
        if count * itemsize != remaining:
            sev = Severity.HIGH if count * itemsize > remaining else Severity.MEDIUM
            ctx.add(
                "LS120",
                sev,
                f".npy data is {remaining} bytes but header implies {count * itemsize}",
                loc,
            )


def _descr_has_object(descr: Any) -> bool:
    if isinstance(descr, str):
        return "O" in descr.lstrip("<>|=")[:1]
    if isinstance(descr, list):
        return any(
            _descr_has_object(item[1])
            for item in descr
            if isinstance(item, tuple) and len(item) >= 2
        )
    return False


def _descr_itemsize(descr: Any) -> int | None:
    if not isinstance(descr, str):
        return None
    code = descr.lstrip("<>|=")
    letter, digits = code[:1], code[1:]
    if letter in "biufcSVmM" and digits.isdigit():
        return int(digits)
    if letter == "U" and digits.isdigit():
        return 4 * int(digits)
    if letter == "?":
        return 1
    return None


def scan_npy_bytes(data: bytes, ctx: Ctx, location: str) -> None:
    scan_npy(io.BytesIO(data), len(data), ctx, location)
