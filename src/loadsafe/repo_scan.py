"""Repository-level checks for Hugging Face-style model folders and the code that loads them."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any

from .findings import Finding, Severity
from .scanners.common import CODE_LIKE, LONG_B64, executable_kind, ssti_hits

CONFIG_FILES = (
    "config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "feature_extractor_config.json",
    "image_processor_config.json",
    "adapter_config.json",
    "sentence_bert_config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "chat_template.json",
)
SCRIPT_EXTS = (
    ".sh",
    ".bash",
    ".zsh",
    ".bat",
    ".cmd",
    ".ps1",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".vbs",
    ".jar",
    ".scr",
    ".msi",
    ".apk",
    ".elf",
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
LICENSE_FILES = re.compile(r"^(licen[cs]e|copying|notice)(\.[a-z]+)?$", re.IGNORECASE)
LOADER_FUNCS = {
    "hf_hub_download",
    "snapshot_download",
    "pipeline",
    "load_dataset",
    "cached_download",
    "hf_hub_url",
}

# (dotted call name) -> (severity, description). Matched on the resolved call name.
DANGEROUS_CALLS: dict[str, tuple[Severity, str]] = {
    "eval": (Severity.HIGH, "eval"),
    "exec": (Severity.HIGH, "exec"),
    "compile": (Severity.MEDIUM, "compile"),
    "__import__": (Severity.HIGH, "__import__"),
    "os.system": (Severity.HIGH, "shell command"),
    "os.popen": (Severity.HIGH, "shell command"),
    "subprocess.run": (Severity.HIGH, "subprocess"),
    "subprocess.Popen": (Severity.HIGH, "subprocess"),
    "subprocess.call": (Severity.HIGH, "subprocess"),
    "subprocess.check_call": (Severity.HIGH, "subprocess"),
    "subprocess.check_output": (Severity.HIGH, "subprocess"),
    "subprocess.getoutput": (Severity.HIGH, "subprocess"),
    "pty.spawn": (Severity.HIGH, "pty spawn"),
    "socket.socket": (Severity.MEDIUM, "raw socket"),
    "socket.create_connection": (Severity.MEDIUM, "raw socket"),
    "urllib.request.urlopen": (Severity.MEDIUM, "network fetch"),
    "requests.get": (Severity.MEDIUM, "network fetch"),
    "requests.post": (Severity.MEDIUM, "network upload"),
    "httpx.get": (Severity.MEDIUM, "network fetch"),
    "httpx.post": (Severity.MEDIUM, "network upload"),
    "marshal.loads": (Severity.HIGH, "marshal bytecode"),
    "pickle.loads": (Severity.MEDIUM, "pickle deserialization"),
    "ctypes.CDLL": (Severity.HIGH, "native library load"),
    "ctypes.cdll.LoadLibrary": (Severity.HIGH, "native library load"),
    "importlib.import_module": (Severity.MEDIUM, "dynamic import"),
    "shutil.rmtree": (Severity.MEDIUM, "recursive delete"),
}
for _fn in (
    "execv",
    "execve",
    "execl",
    "execlp",
    "execvp",
    "spawnl",
    "spawnv",
    "spawnve",
    "posix_spawn",
    "fork",
    "forkpty",
    "remove",
    "unlink",
):
    DANGEROUS_CALLS[f"os.{_fn}"] = (
        Severity.HIGH if not _fn.startswith(("remove", "unlink")) else Severity.MEDIUM,
        "process/file operation",
    )
UNSAFE_LOADERS = {
    "pickle.load": "pickle.load",
    "pickle.loads": "pickle.loads",
    "joblib.load": "joblib.load",
    "dill.load": "dill.load",
    "dill.loads": "dill.loads",
    "cloudpickle.load": "cloudpickle.load",
    "pd.read_pickle": "pandas.read_pickle",
    "pandas.read_pickle": "pandas.read_pickle",
}


def _dotted(node: ast.AST, aliases: dict[str, str]) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(aliases.get(node.id, node.id))
        return ".".join(reversed(parts))
    return None


def _kw(call: ast.Call, name: str) -> ast.expr | None:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _is_const(node: ast.expr | None, value: Any) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


class _CodeVisitor(ast.NodeVisitor):
    def __init__(self, rel: str, findings: list[Finding]) -> None:
        self.rel = rel
        self.findings = findings
        self.aliases: dict[str, str] = {}
        self.depth = 0

    def _add(self, rule: str, sev: Severity, msg: str, node: ast.AST, **d: Any) -> None:
        self.findings.append(
            Finding(rule, sev, msg, self.rel, f"line {getattr(node, 'lineno', 0)}", d)
        )

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.aliases[a.asname or a.name.split(".")[0]] = (
                a.name if a.asname else a.name.split(".")[0]
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            for a in node.names:
                self.aliases[a.asname or a.name] = f"{node.module}.{a.name}"

    def _scoped(self, node: ast.AST) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_FunctionDef = _scoped  # type: ignore[assignment]
    visit_AsyncFunctionDef = _scoped  # type: ignore[assignment]
    visit_Lambda = _scoped  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func, self.aliases) or ""
        short = name.rsplit(".", 1)[-1]
        if name in DANGEROUS_CALLS:
            sev, desc = DANGEROUS_CALLS[name]
            if self.depth == 0 and sev >= Severity.HIGH:
                sev = Severity.CRITICAL
                desc += " at import time"
            self._add("LS201", sev, f"{name}() call ({desc})", node, call=name)
        # --- consumer-side loader hygiene -------------------------------------------------
        if short == "from_pretrained" or name.rsplit(".", 1)[-1] in LOADER_FUNCS:
            trc = _kw(node, "trust_remote_code")
            if _is_const(trc, True):
                self._add("LS211", Severity.HIGH, f"{short}(trust_remote_code=True)", node)
            first = (
                node.args[0]
                if node.args
                else _kw(node, "repo_id")
                or _kw(node, "pretrained_model_name_or_path")
                or _kw(node, "model")
            )
            local = (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and (first.value.startswith((".", "/", "~")) or os.sep in first.value[:1])
            )
            if short != "hf_hub_url" and not local and first is not None:
                rev = _kw(node, "revision")
                if rev is None:
                    self._add("LS210", Severity.MEDIUM, f"{short}() without revision= pin", node)
                elif (
                    isinstance(rev, ast.Constant)
                    and isinstance(rev.value, str)
                    and not SHA_RE.match(rev.value)
                ):
                    self._add(
                        "LS210",
                        Severity.LOW,
                        f"{short}() pinned to mutable ref {rev.value!r}, not a commit sha",
                        node,
                    )
        if name in ("torch.load",) or (short == "load" and name.endswith("torch.load")):
            wo = _kw(node, "weights_only")
            if _is_const(wo, False):
                self._add("LS212", Severity.HIGH, "torch.load(weights_only=False)", node)
            elif wo is None:
                self._add(
                    "LS212",
                    Severity.LOW,
                    "torch.load() without weights_only=True (unsafe default before torch 2.6)",
                    node,
                )
        elif name in UNSAFE_LOADERS:
            self._add(
                "LS212",
                Severity.MEDIUM,
                f"{UNSAFE_LOADERS[name]}() deserializes untrusted data",
                node,
            )
        elif name in ("numpy.load", "np.load") and _is_const(_kw(node, "allow_pickle"), True):
            self._add("LS212", Severity.MEDIUM, "np.load(allow_pickle=True)", node)
        self.generic_visit(node)


def scan_python(path: Path, rel: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=rel)
    except (SyntaxError, ValueError) as exc:
        return [
            Finding(
                "LS201", Severity.MEDIUM, f"Python file could not be parsed: {exc}", rel, None, {}
            )
        ]
    _CodeVisitor(rel, findings).visit(tree)
    if LONG_B64.search(src) and ("b64decode" in src or "exec" in src):
        findings.append(
            Finding(
                "LS201",
                Severity.HIGH,
                "long encoded blob next to b64decode/exec (obfuscated payload pattern)",
                rel,
                None,
                {},
            )
        )
    return findings


def _walk_strings(obj: Any, path: str = "") -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:2000]):
            yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def _check_config(path: Path, rel: str, findings: list[Finding]) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > (16 << 20):
            return None
        cfg = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (ValueError, OSError):
        findings.append(Finding("LS205", Severity.LOW, "config is not valid JSON", rel, None, {}))
        return None
    if not isinstance(cfg, dict):
        return None
    auto_map = cfg.get("auto_map")
    if auto_map:
        refs = []
        for v in auto_map.values() if isinstance(auto_map, dict) else []:
            refs.extend(v if isinstance(v, list) else [v])
        refs = [r for r in refs if isinstance(r, str)]
        findings.append(
            Finding(
                "LS200",
                Severity.HIGH,
                "auto_map requires trust_remote_code: " + ", ".join(refs[:6]),
                rel,
                "auto_map",
                {"auto_map": refs},
            )
        )
        for r in refs:
            if "--" in r:
                findings.append(
                    Finding(
                        "LS205",
                        Severity.HIGH,
                        f"auto_map pulls code from another repository: {r}",
                        rel,
                        "auto_map",
                        {},
                    )
                )
    if cfg.get("custom_pipelines"):
        findings.append(
            Finding(
                "LS200",
                Severity.HIGH,
                "custom_pipelines declares repository code",
                rel,
                "custom_pipelines",
                {},
            )
        )
    if cfg.get("trust_remote_code") is True:
        findings.append(
            Finding(
                "LS200",
                Severity.HIGH,
                "config sets trust_remote_code=true",
                rel,
                "trust_remote_code",
                {},
            )
        )
    templates: list[tuple[str, str]] = []
    ct = cfg.get("chat_template")
    if isinstance(ct, str):
        templates.append(("chat_template", ct))
    elif isinstance(ct, list):
        for i, t in enumerate(ct):
            if isinstance(t, dict) and isinstance(t.get("template"), str):
                templates.append((f"chat_template[{t.get('name', i)}]", t["template"]))
    for key, tpl in templates:
        hits = ssti_hits(tpl)
        if hits:
            findings.append(
                Finding(
                    "LS141",
                    Severity.HIGH,
                    "chat template contains template-injection patterns: " + ", ".join(hits),
                    rel,
                    key,
                    {"excerpt": tpl[:200]},
                )
            )
    for jpath, s in _walk_strings({k: v for k, v in cfg.items() if k != "chat_template"}):
        if CODE_LIKE.search(s):
            findings.append(
                Finding(
                    "LS205",
                    Severity.MEDIUM,
                    "field contains code-like text",
                    rel,
                    jpath,
                    {"excerpt": s[:160]},
                )
            )
        elif LONG_B64.search(s):
            findings.append(
                Finding("LS205", Severity.LOW, "field contains a long encoded blob", rel, jpath, {})
            )
    return cfg


def _has_license(root: Path, files: list[Path]) -> bool:
    if any(LICENSE_FILES.match(p.name) and p.parent == root for p in files):
        return True
    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")[:20000]
        if text.startswith("---"):
            front = text.split("---", 2)[1] if text.count("---") >= 2 else ""
            if re.search(r"^license(_name|_link)?\s*:\s*\S", front, re.MULTILINE):
                return True
    return False


def scan_repo(root: Path, files: list[Path], looks_like_model: bool) -> list[Finding]:
    findings: list[Finding] = []
    root_res = root.resolve()
    for p in files:
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            try:
                target = p.resolve()
                if root_res not in target.parents and target != root_res:
                    findings.append(
                        Finding(
                            "LS204",
                            Severity.HIGH,
                            f"symlink points outside the repository: {os.readlink(p)}",
                            rel,
                            None,
                            {},
                        )
                    )
                    continue
            except OSError:
                continue
        if not p.is_file():
            continue
        name = p.name
        low = name.lower()
        if low.endswith(".py"):
            findings.extend(scan_python(p, rel))
            continue
        if name in CONFIG_FILES:
            _check_config(p, rel, findings)
            continue
        if low.endswith(".jinja") or low == "chat_template.jinja":
            hits = ssti_hits(p.read_text(encoding="utf-8", errors="replace"))
            if hits:
                findings.append(
                    Finding(
                        "LS141",
                        Severity.HIGH,
                        "chat template contains template-injection patterns: " + ", ".join(hits),
                        rel,
                        None,
                        {},
                    )
                )
            continue
        try:
            with open(p, "rb") as f:
                head = f.read(64)
        except OSError:
            continue
        exe = executable_kind(head)
        if low.endswith(SCRIPT_EXTS) or (exe and not low.endswith((".py", ".txt", ".md"))):
            findings.append(
                Finding(
                    "LS202",
                    Severity.HIGH,
                    f"{exe or 'script/binary'} in model repository",
                    rel,
                    None,
                    {},
                )
            )
    py_files = [p for p in files if p.suffix == ".py"]
    if looks_like_model and py_files and not any(f.rule_id == "LS200" for f in findings):
        findings.append(
            Finding(
                "LS200",
                Severity.MEDIUM,
                f"{len(py_files)} Python file(s) shipped with the model (custom code)",
                ".",
                None,
                {"files": [p.relative_to(root).as_posix() for p in py_files][:20]},
            )
        )
    if looks_like_model and not _has_license(root, files):
        findings.append(
            Finding("LS203", Severity.LOW, "no LICENSE file or model-card license", ".", None, {})
        )
    return findings
