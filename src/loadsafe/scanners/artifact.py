"""Format detection and per-file dispatch."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import lzma
import mmap
import os
import struct
import time
import zlib
from pathlib import Path
from typing import IO

from ..findings import ArtifactResult, Severity
from ..pickle_scan import scan_pickle
from ..policy import Policy
from .common import Ctx, executable_kind
from .gguf_scan import scan_gguf
from .keras_scan import scan_hdf5_model_config
from .onnx_scan import scan_onnx
from .tensors import scan_npy, scan_safetensors
from .zip_scan import scan_zip

MODEL_EXTENSIONS = frozenset(
    {
        ".pt",
        ".pth",
        ".bin",
        ".ckpt",
        ".pkl",
        ".pickle",
        ".joblib",
        ".jbl",
        ".dill",
        ".npy",
        ".npz",
        ".safetensors",
        ".onnx",
        ".gguf",
        ".h5",
        ".hdf5",
        ".keras",
        ".mar",
        ".model",
        ".sav",
        ".dat",
    }
)
PICKLE_EXTS = frozenset({".pkl", ".pickle", ".joblib", ".jbl", ".dill", ".sav", ".dat"})
TORCH_LEGACY_MAGIC = b"\x80\x02\x8a\x0al\xfc\x9cF\xf9 j\xa8P\x19."
PICKLE_TEXT_START = frozenset(b"(c]}).NIVSldKFLGJMTUXa")
PICKLE_BASED = frozenset(
    {"pickle", "pytorch-legacy", "pytorch-zip", "torchscript", "compressed-pickle"}
)
NON_PICKLE_PROMISES = {
    ".safetensors": {"safetensors"},
    ".gguf": {"gguf"},
    ".onnx": {"onnx"},
    ".h5": {"hdf5"},
    ".hdf5": {"hdf5"},
    ".npy": {"npy"},
    ".keras": {"keras-v3"},
    ".npz": {"npz", "zip"},
}
MAX_DECOMPRESSED = 4 << 30
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def sniff(head: bytes, ext: str, size: int) -> str:
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return "zip"
    if head.startswith(b"\x93NUMPY"):
        return "npy"
    if head.startswith(b"GGUF"):
        return "gguf"
    if head.startswith(HDF5_MAGIC):
        return "hdf5"
    if head.startswith(TORCH_LEGACY_MAGIC):
        return "pytorch-legacy"
    if len(head) >= 2 and head[0] == 0x80 and head[1] <= 5:
        return "pickle"
    if len(head) >= 9 and head[8:9] == b"{":
        (n,) = struct.unpack("<Q", head[:8])
        if 0 < n <= size - 8:
            return "safetensors"
    if ext == ".onnx":
        return "onnx"
    if ext in PICKLE_EXTS or ext in (".gz", ".z", ".bz2", ".xz", ".lzma"):
        if (
            head.startswith(b"\x1f\x8b")
            or head.startswith(b"BZh")
            or head.startswith(b"\xfd7zXZ\x00")
            or head.startswith(b"\x5d\x00\x00")
            or (len(head) > 1 and head[0] == 0x78 and head[1] in (0x01, 0x5E, 0x9C, 0xDA))
        ):
            return "compressed-pickle"
        if head and head[0] in PICKLE_TEXT_START:
            return "pickle"
    if executable_kind(head):
        return "executable"
    return "unknown"


def _decompress(f: IO[bytes], head: bytes) -> IO[bytes]:
    if head.startswith(b"\x1f\x8b"):
        return gzip.GzipFile(fileobj=f)  # type: ignore[return-value]
    if head.startswith(b"BZh"):
        return bz2.BZ2File(f)
    if head.startswith(b"\xfd7zXZ\x00") or head.startswith(b"\x5d\x00\x00"):
        return lzma.LZMAFile(f)
    d = zlib.decompressobj()
    out = io.BytesIO()
    while True:
        chunk = f.read(1 << 20)
        if not chunk:
            break
        out.write(d.decompress(chunk, MAX_DECOMPRESSED - out.tell() + 1))
        if out.tell() > MAX_DECOMPRESSED:
            raise ValueError("decompressed size limit exceeded")
    out.write(d.flush())
    out.seek(0)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def is_candidate(path: Path) -> bool:
    if path.suffix.lower() in MODEL_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return False
    fmt = sniff(head, path.suffix.lower(), path.stat().st_size)
    return fmt in ("zip", "npy", "gguf", "hdf5", "pickle", "pytorch-legacy", "safetensors") and (
        path.suffix.lower() not in (".json", ".txt", ".md", ".py", ".zip")
    )


def scan_file(
    path: Path, display: str, policy: Policy | None = None, hash_file: bool = True
) -> ArtifactResult:
    policy = policy or Policy()
    t0 = time.perf_counter()
    size = path.stat().st_size
    ext = path.suffix.lower()
    ctx = Ctx(display, policy)
    with open(path, "rb") as f:
        head = f.read(64)
        f.seek(0)
        fmt = sniff(head, ext, size)
        try:
            fmt = _dispatch(fmt, f, size, head, ctx, ext)
        except MemoryError:
            ctx.add("LS299", Severity.MEDIUM, "out of memory while scanning")
        except Exception as exc:  # noqa: BLE001 - never crash a CI run on one bad file
            ctx.add("LS299", Severity.MEDIUM, f"scanner error: {type(exc).__name__}: {exc}")
    promised = NON_PICKLE_PROMISES.get(ext)
    if promised and fmt not in promised:
        sev = Severity.HIGH if fmt in PICKLE_BASED or fmt == "executable" else Severity.MEDIUM
        ctx.add("LS160", sev, f"extension {ext} but content looks like {fmt}")
    elif ext in PICKLE_EXTS | {".pt", ".pth", ".ckpt"} and fmt == "executable":
        ctx.add("LS160", Severity.HIGH, f"extension {ext} but content is an executable")
    if fmt == "executable":
        ctx.add("LS202", Severity.HIGH, f"{executable_kind(head)} where a model file was expected")
    scan_seconds = time.perf_counter() - t0
    digest = ""
    if hash_file:
        t1 = time.perf_counter()
        digest = sha256_file(path)
        ctx.metadata["hash_seconds"] = round(time.perf_counter() - t1, 6)
    return ArtifactResult(display, fmt, size, digest, ctx.findings, ctx.metadata, scan_seconds)


def _dispatch(fmt: str, f: IO[bytes], size: int, head: bytes, ctx: Ctx, ext: str) -> str:
    if fmt == "zip":
        return scan_zip(f, ctx)
    if fmt == "npy":
        scan_npy(f, size, ctx)
    elif fmt == "gguf":
        scan_gguf(f, size, ctx)
    elif fmt == "safetensors":
        scan_safetensors(f, size, ctx)
    elif fmt == "hdf5":
        if size:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                scan_hdf5_model_config(mm[:], ctx) if size < (512 << 20) else _hdf5_big(mm, ctx)
    elif fmt == "onnx":
        if size:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                scan_onnx(mm, ctx)
        else:
            scan_onnx(b"", ctx)
    elif fmt in ("pickle", "pytorch-legacy", "compressed-pickle"):
        stream: IO[bytes] = _decompress(f, head) if fmt == "compressed-pickle" else f
        res = scan_pickle(stream, ctx.path, "", ctx.policy, multi=True)
        ctx.findings.extend(res.findings)
        ctx.metadata.update(
            pickle_opcodes=res.opcodes, pickles=res.pickles, pickle_globals=res.globals
        )
        if fmt == "pickle" and res.complete and res.bytes_consumed < size and not res.pickles > 1:
            ctx.metadata["trailing_bytes"] = size - res.bytes_consumed
        ctx.add("LS107", Severity.LOW, f"{fmt} artifact is deserialized with pickle")
    return fmt


def _hdf5_big(mm: mmap.mmap, ctx: Ctx) -> None:
    # Only the first 512 MiB are searched for configs in very large HDF5 files.
    scan_hdf5_model_config(mm[: 512 << 20], ctx)
    ctx.metadata["hdf5_partial_scan"] = True


def iter_files(root: Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            d for d in dirnames if d not in (".git", "__pycache__", ".venv", "node_modules")
        ]
        for fn in filenames:
            out.append(Path(dirpath) / fn)
    return sorted(out)
