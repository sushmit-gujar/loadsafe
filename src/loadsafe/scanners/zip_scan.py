"""ZIP-container checks and member dispatch (PyTorch, Keras v3, NumPy .npz)."""

from __future__ import annotations

import json
import posixpath
import struct
import zipfile
from typing import IO

from ..findings import Severity
from ..pickle_scan import scan_pickle
from .common import Ctx, executable_kind
from .keras_scan import check_keras_config, scan_hdf5_model_config
from .tensors import scan_npy_bytes

MAX_MEMBER = 2 << 30
SCRIPT_EXTS = (
    ".py",
    ".sh",
    ".bash",
    ".bat",
    ".cmd",
    ".ps1",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".js",
    ".vbs",
)


def zip_kind(names: list[str]) -> str:
    if any(n == "data.pkl" or n.endswith("/data.pkl") for n in names):
        if any("/code/" in n or n.startswith("code/") for n in names) or any(
            n.endswith("constants.pkl") for n in names
        ):
            return "torchscript"
        return "pytorch-zip"
    if "config.json" in names and ("model.weights.h5" in names or "metadata.json" in names):
        return "keras-v3"
    if names and all(n.endswith(".npy") for n in names):
        return "npz"
    return "zip"


def _structure_checks(f: IO[bytes], zf: zipfile.ZipFile, ctx: Ctx) -> None:
    seen: set[str] = set()
    for info in zf.infolist():
        name = info.filename
        if name in seen:
            ctx.add(
                "LS110",
                Severity.HIGH,
                "duplicate member name: loaders and scanners may read different copies",
                name,
            )
        seen.add(name)
        norm = posixpath.normpath(name)
        if name.startswith("/") or norm.startswith("../") or norm == ".." or ":" in name[:3]:
            ctx.add("LS110", Severity.HIGH, "member path escapes the archive root", name)
        if info.flag_bits & 0x1:
            ctx.add("LS110", Severity.MEDIUM, "member is flagged as encrypted", name)
        # Compare central-directory name with the local file header name.
        try:
            f.seek(info.header_offset)
            hdr = f.read(30)
            if len(hdr) == 30 and hdr[:4] == b"PK\x03\x04":
                n_len, x_len = struct.unpack("<HH", hdr[26:30])
                local = f.read(n_len)
                cd_raw = info.orig_filename.encode(
                    "utf-8" if info.flag_bits & 0x800 else "cp437", errors="replace"
                )
                if local != cd_raw:
                    ctx.add(
                        "LS110",
                        Severity.HIGH,
                        f"local header name {local[:80]!r} differs from central directory name",
                        name,
                    )
            else:
                ctx.add(
                    "LS110",
                    Severity.HIGH,
                    "central directory points to an invalid local header",
                    name,
                )
        except OSError:
            ctx.add("LS110", Severity.MEDIUM, "could not read local header", name)


def scan_zip(f: IO[bytes], ctx: Ctx) -> str:
    try:
        zf = zipfile.ZipFile(f)
    except (zipfile.BadZipFile, OSError) as exc:
        ctx.add("LS110", Severity.HIGH, f"corrupt ZIP container: {exc}")
        return "zip"
    with zf:
        names = zf.namelist()
        kind = zip_kind(names)
        ctx.metadata["members"] = len(names)
        _structure_checks(f, zf, ctx)
        pickles = 0
        for info in zf.infolist():
            name = info.filename
            if info.is_dir():
                continue
            low = name.lower()
            base = posixpath.basename(low)
            is_pickle = low.endswith((".pkl", ".pickle"))
            in_code = "/code/" in name or name.startswith("code/")
            if info.file_size > MAX_MEMBER and (is_pickle or low.endswith((".npy", ".json"))):
                ctx.add("LS299", Severity.MEDIUM, "member too large to scan", name)
                continue
            if is_pickle:
                pickles += 1
                try:
                    data = zf.read(info)
                except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError) as exc:
                    ctx.add("LS110", Severity.HIGH, f"member unreadable: {exc}", name)
                    continue
                res = scan_pickle(data, ctx.path, name, ctx.policy, multi=False)
                ctx.findings.extend(res.findings)
                ctx.metadata.setdefault("pickle_globals", {}).update(res.globals)
                ctx.metadata["pickle_opcodes"] = ctx.metadata.get("pickle_opcodes", 0) + res.opcodes
                if base not in ("data.pkl", "constants.pkl") and kind in (
                    "pytorch-zip",
                    "torchscript",
                ):
                    ctx.add(
                        "LS111",
                        Severity.MEDIUM,
                        "additional pickle member in PyTorch archive",
                        name,
                    )
                continue
            if kind == "torchscript" and in_code and low.endswith(".py"):
                ctx.metadata["torchscript_code_files"] = (
                    ctx.metadata.get("torchscript_code_files", 0) + 1
                )
                continue
            if low.endswith(".npy") and kind == "npz":
                scan_npy_bytes(zf.read(info), ctx, name)
                continue
            if kind == "keras-v3" and name == "config.json":
                try:
                    cfg = json.loads(zf.read(info))
                except ValueError as exc:
                    ctx.add("LS120", Severity.MEDIUM, f"config.json is not valid JSON: {exc}", name)
                    continue
                ctx.metadata["keras_layers"] = check_keras_config(cfg, ctx, name)
                continue
            if kind == "keras-v3" and low.endswith(".h5") and info.file_size < MAX_MEMBER:
                scan_hdf5_model_config(zf.read(info), ctx)
                continue
            with zf.open(info) as m:
                head = m.read(64)
            exe = executable_kind(head)
            if exe or low.endswith(SCRIPT_EXTS):
                ctx.add("LS111", Severity.HIGH, f"archive contains {exe or 'a script'}", name)
        if kind == "torchscript" and ctx.metadata.get("torchscript_code_files"):
            ctx.add(
                "LS112",
                Severity.MEDIUM,
                f"TorchScript archive with "
                f"{ctx.metadata['torchscript_code_files']} code files; torch.jit.load "
                "executes them",
            )
        if kind in ("pytorch-zip", "torchscript") and pickles:
            ctx.add(
                "LS107",
                Severity.LOW,
                "PyTorch checkpoint is pickle-based; load with "
                "torch.load(weights_only=True) or convert to safetensors",
            )
        if kind == "pytorch-zip" and not pickles:
            ctx.add("LS110", Severity.MEDIUM, "PyTorch archive without data.pkl")
    return kind
