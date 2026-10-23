"""Keras model config checks (.keras zip config.json and legacy HDF5 model_config)."""

from __future__ import annotations

import json
import re
from typing import Any

from ..findings import Severity
from .common import Ctx

ALLOWED_MODULE_PREFIXES = (
    "keras",
    "tensorflow",
    "tf_keras",
    "keras_core",
    "keras_nlp",
    "keras_hub",
    "keras_cv",
    "numpy",
    "builtins",
)


def _walk(obj: Any, path: str = "$") -> Any:
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def check_keras_config(config: Any, ctx: Ctx, where: str) -> int:
    """Return the number of layers seen; add Lambda / module findings to ``ctx``."""
    layers = 0
    for jpath, node in _walk(config):
        if not isinstance(node, dict):
            continue
        cls = node.get("class_name")
        if isinstance(cls, str) and isinstance(node.get("config"), dict):
            layers += 1
        if cls == "Lambda":
            cfg = node.get("config") or {}
            fn = cfg.get("function") if isinstance(cfg, dict) else None
            marshalled = False
            if isinstance(fn, dict):
                inner = fn.get("config")
                marshalled = fn.get("class_name") == "__lambda__" or (
                    isinstance(inner, dict) and "code" in inner
                )
            elif isinstance(fn, list) and fn and isinstance(fn[0], str):
                marshalled = True  # Keras 2: [base64 marshal bytecode, defaults, closure]
            elif isinstance(fn, str) and cfg.get("function_type") == "lambda":
                marshalled = True
            sev = Severity.CRITICAL if marshalled else Severity.HIGH
            desc = "with serialized Python bytecode" if marshalled else "referencing a function"
            ctx.add(
                "LS150",
                sev,
                f"Keras Lambda layer {desc} ({cfg.get('name', '?')})",
                f"{where}:{jpath}",
            )
        module = node.get("module")
        if (
            isinstance(module, str)
            and isinstance(cls, str)
            and not module.startswith(ALLOWED_MODULE_PREFIXES)
        ):
            ctx.add(
                "LS151",
                Severity.HIGH,
                f"config object imports {module}.{cls}",
                f"{where}:{jpath}",
                module=module,
                class_name=cls,
            )
        fn_name = node.get("function_name") or node.get("fn")
        if isinstance(fn_name, str) and re.match(r"^(os|subprocess|builtins|posix)\.", fn_name):
            ctx.add(
                "LS151", Severity.HIGH, f"config references function {fn_name}", f"{where}:{jpath}"
            )
    return layers


_JSON_START = re.compile(rb'\{\s*"class_name"\s*:')


def scan_hdf5_model_config(data: bytes, ctx: Ctx) -> None:
    """Heuristic: find JSON model configs stored as HDF5 attributes without h5py."""
    found = 0
    decoder = json.JSONDecoder()
    for m in _JSON_START.finditer(data):
        if found >= 4:
            break
        chunk = data[m.start() : m.start() + (64 << 20)].decode("latin-1")
        try:
            obj, _ = decoder.raw_decode(chunk)
        except ValueError:
            continue
        found += 1
        ctx.metadata["keras_layers"] = check_keras_config(obj, ctx, "hdf5:model_config")
    if not found and (b'"class_name": "Lambda"' in data or b'"class_name":"Lambda"' in data):
        ctx.add(
            "LS150", Severity.HIGH, "Lambda layer marker found in HDF5 file (config not parseable)"
        )
    ctx.metadata["keras_config_found"] = bool(found)
