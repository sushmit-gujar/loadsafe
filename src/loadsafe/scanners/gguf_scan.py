"""GGUF header and metadata validation (no llama.cpp / gguf package required)."""

from __future__ import annotations

import struct
from typing import IO, Any

from ..findings import Severity
from .common import Ctx, ssti_hits

# ggml type id -> (block size, bytes per block); from ggml.h / gguf-py GGML_QUANT_SIZES.
GGML_TYPE_SIZES: dict[int, tuple[int, int]] = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 40),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 50),
    20: (32, 18),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    24: (1, 1),
    25: (1, 2),
    26: (1, 4),
    27: (1, 8),
    28: (1, 8),
    29: (256, 56),
    30: (1, 2),
    34: (256, 54),
    35: (256, 66),
    39: (32, 17),
    40: (64, 36),
    41: (128, 18),
}
_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
MAX_KV = 1 << 20
MAX_TENSORS = 1 << 20
MAX_DIMS = 4  # GGML_MAX_DIMS
MAX_ARRAY = 1 << 26


class _Bad(Exception):
    pass


class _Reader:
    def __init__(self, f: IO[bytes], size: int, endian: str, version: int) -> None:
        self.f, self.size, self.e, self.version = f, size, endian, version

    def remaining(self) -> int:
        return self.size - self.f.tell()

    def read(self, n: int) -> bytes:
        if n < 0 or n > self.remaining():
            raise _Bad(f"read of {n} bytes past end of file at {self.f.tell()}")
        return self.f.read(n)

    def scalar(self, fmt: str) -> Any:
        b = self.read(struct.calcsize(fmt))
        return struct.unpack(self.e + fmt, b)[0]

    def count(self) -> int:
        return int(self.scalar("I" if self.version == 1 else "Q"))

    def string(self) -> str:
        n = self.count()
        if n > self.remaining():
            raise _Bad(f"string length {n} exceeds file at {self.f.tell()}")
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int, depth: int = 0) -> Any:
        if vtype in _SCALARS:
            return self.scalar(_SCALARS[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            if depth > 2:
                raise _Bad("nested arrays too deep")
            etype = self.scalar("I")
            n = self.count()
            min_size = struct.calcsize(_SCALARS[etype]) if etype in _SCALARS else 8
            if n > MAX_ARRAY or n * min_size > self.remaining():
                raise _Bad(f"array of {n} elements exceeds file")
            if etype in _SCALARS and etype != 7:
                fmt = _SCALARS[etype]
                self.read(n * struct.calcsize(fmt))  # skip numeric arrays (vocab scores etc.)
                return f"<array {n} x type {etype}>"
            return [self.value(etype, depth + 1) for _ in range(n)]
        raise _Bad(f"unknown GGUF value type {vtype}")


def scan_gguf(f: IO[bytes], size: int, ctx: Ctx) -> None:
    magic = f.read(4)
    if magic != b"GGUF":
        ctx.add("LS140", Severity.HIGH, "bad GGUF magic")
        return
    raw_ver = f.read(4)
    if len(raw_ver) < 4:
        ctx.add("LS140", Severity.HIGH, "truncated GGUF header")
        return
    endian = "<"
    version = struct.unpack("<I", raw_ver)[0]
    if version > 0xFFFF:
        endian, version = ">", struct.unpack(">I", raw_ver)[0]
    ctx.metadata["gguf_version"] = version
    if version not in (1, 2, 3):
        ctx.add("LS140", Severity.HIGH, f"unsupported GGUF version {version}")
        return
    r = _Reader(f, size, endian, version)
    try:
        n_tensors = r.count()
        n_kv = r.count()
        ctx.metadata.update(tensors=n_tensors, kv=n_kv)
        if n_tensors > MAX_TENSORS or n_kv > MAX_KV:
            ctx.add(
                "LS140",
                Severity.HIGH,
                f"implausible counts: {n_tensors} tensors, {n_kv} "
                "metadata entries (allocation/overflow attack surface)",
            )
            return
        alignment = 32
        for _ in range(n_kv):
            key = r.string()
            vtype = r.scalar("I")
            val = r.value(vtype)
            if key == "general.alignment" and isinstance(val, int):
                alignment = val
                if val <= 0 or val & (val - 1):
                    ctx.add(
                        "LS140",
                        Severity.HIGH,
                        f"general.alignment {val} is not a power of two",
                        key,
                    )
                    alignment = 32
            elif key.startswith("tokenizer.chat_template") and isinstance(val, str):
                hits = ssti_hits(val)
                ctx.metadata.setdefault("chat_templates", []).append(key)
                if hits:
                    ctx.add(
                        "LS141",
                        Severity.HIGH,
                        "chat template contains template-injection patterns: " + ", ".join(hits),
                        key,
                        excerpt=val[:200],
                    )
            elif key in ("general.architecture", "general.name", "general.license"):
                ctx.metadata[key] = str(val)[:200]
        ends: list[int] = []
        for i in range(n_tensors):
            name = r.string()
            n_dims = r.scalar("I")
            if n_dims > MAX_DIMS:
                ctx.add(
                    "LS140",
                    Severity.HIGH,
                    f"tensor has {n_dims} dims (max {MAX_DIMS})",
                    f"tensor {name}",
                )
                return
            dims = [r.count() for _ in range(n_dims)]
            ttype = r.scalar("I")
            offset = r.scalar("Q")
            elems = 1
            for d in dims:
                elems *= d
            if elems >= 1 << 63:
                ctx.add(
                    "LS140",
                    Severity.HIGH,
                    f"tensor element count overflows int64 {dims}",
                    f"tensor {name}",
                )
                continue
            if ttype not in GGML_TYPE_SIZES:
                ctx.add(
                    "LS140", Severity.MEDIUM, f"unknown ggml tensor type {ttype}", f"tensor {name}"
                )
                continue
            block, bsize = GGML_TYPE_SIZES[ttype]
            if dims and dims[0] % block:
                ctx.add(
                    "LS140",
                    Severity.HIGH,
                    f"row size {dims[0]} not a multiple of block size {block}",
                    f"tensor {name}",
                )
                continue
            if offset % alignment:
                ctx.add(
                    "LS140",
                    Severity.MEDIUM,
                    f"tensor offset {offset} not aligned to {alignment}",
                    f"tensor {name}",
                )
            ends.append(offset + elems // block * bsize)
            if i == 0:
                ctx.metadata["first_tensor"] = name
        header_end = f.tell()
        data_start = header_end + (-header_end % alignment)
        if ends and data_start + max(ends) > size:
            ctx.add(
                "LS140",
                Severity.HIGH,
                f"tensor data extends {data_start + max(ends) - size} bytes past end of file",
            )
    except _Bad as exc:
        ctx.add("LS140", Severity.HIGH, f"malformed GGUF: {exc}")
    except (struct.error, UnicodeDecodeError) as exc:
        ctx.add("LS140", Severity.HIGH, f"malformed GGUF: {exc}")
