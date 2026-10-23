"""Rule catalogue with OWASP Top 10 for LLM Applications (2025) and MITRE ATLAS mappings.

Only mappings we consider accurate are listed. Framework references:

* OWASP LLM03:2025 Supply Chain, LLM04:2025 Data and Model Poisoning
  (https://genai.owasp.org/llm-top-10/)
* MITRE ATLAS (https://atlas.mitre.org/):
  AML.T0010.001 AI Supply Chain Compromise: AI Software,
  AML.T0010.003 AI Supply Chain Compromise: Model,
  AML.T0011.000 User Execution: Unsafe AI Artifacts,
  AML.T0018.002 Manipulate AI Model: Embed Malware.
"""

from __future__ import annotations

from typing import Any

from .findings import Rule

SUPPLY = "LLM03:2025 Supply Chain"
POISON = "LLM04:2025 Data and Model Poisoning"

T_SOFTWARE = "AML.T0010.001"
T_MODEL = "AML.T0010.003"
T_UNSAFE = "AML.T0011.000"
T_MALWARE = "AML.T0018.002"

ATLAS_NAMES = {
    T_SOFTWARE: "AI Supply Chain Compromise: AI Software",
    T_MODEL: "AI Supply Chain Compromise: Model",
    T_UNSAFE: "User Execution: Unsafe AI Artifacts",
    T_MALWARE: "Manipulate AI Model: Embed Malware",
}

_PAYLOAD: dict[str, Any] = dict(owasp_llm=(SUPPLY, POISON), atlas=(T_MODEL, T_UNSAFE, T_MALWARE))
_ARTIFACT: dict[str, Any] = dict(owasp_llm=(SUPPLY,), atlas=(T_MODEL, T_UNSAFE))
_STRUCT: dict[str, Any] = dict(owasp_llm=(SUPPLY,), atlas=(T_MODEL,))
_CODE: dict[str, Any] = dict(owasp_llm=(SUPPLY,), atlas=(T_SOFTWARE, T_UNSAFE))

_RULES = [
    # --- pickle --------------------------------------------------------------------------
    Rule(
        "LS100",
        "Dangerous callable invoked during unpickling",
        "The pickle stream calls (REDUCE/INST/OBJ/NEWOBJ) a callable that can execute code, "
        "spawn processes, touch the network or the filesystem. Loading this file runs it.",
        cwe=("CWE-502",),
        help_text="Do not load. Obtain the model in safetensors or rebuild "
        "it from a trusted source.",
        **_PAYLOAD,
    ),
    Rule(
        "LS101",
        "Dangerous global imported by pickle",
        "The pickle stream resolves a dangerous global (GLOBAL/STACK_GLOBAL/INST) even if no "
        "direct call was observed; it can still be invoked via BUILD/__setstate__ or gadgets.",
        cwe=("CWE-502",),
        help_text="Treat as malicious unless the import is explained.",
        **_PAYLOAD,
    ),
    Rule(
        "LS102",
        "Non-allowlisted global imported by pickle",
        "The pickle imports a class or function outside loadsafe's allowlist of known-safe ML "
        "globals. Unknown callables are how scanner bypasses happen.",
        cwe=("CWE-502",),
        help_text="Review the import; add it to an allowlist if expected.",
        **_ARTIFACT,
    ),
    Rule(
        "LS103",
        "Dynamically resolved or indirect callable",
        "A STACK_GLOBAL operand or a called object is not a constant (built from memo "
        "tricks, getattr chains or other call results): a common obfuscation pattern.",
        cwe=("CWE-502",),
        **_PAYLOAD,
    ),
    Rule(
        "LS104",
        "Malformed or truncated pickle stream",
        "The stream could not be fully disassembled. pickle executes opcodes as it reads, so "
        "a payload placed before the corruption still runs while scanners may give up.",
        cwe=("CWE-502", "CWE-20"),
        **_ARTIFACT,
    ),
    Rule(
        "LS105",
        "Nested pickle embedded in a pickle",
        "A bytes constant inside the pickle is itself a pickle stream (e.g. passed to "
        "pickle.loads). Its findings are reported with a nested location.",
        cwe=("CWE-502",),
        **_PAYLOAD,
    ),
    Rule(
        "LS106",
        "Pickle extension-registry opcode",
        "EXT1/EXT2/EXT4 resolve globals through copyreg's extension registry, hiding the "
        "imported name from naive scanners.",
        cwe=("CWE-502",),
        **_ARTIFACT,
    ),
    Rule(
        "LS107",
        "Pickle-based serialization format",
        "The artifact is deserialized with pickle, which can execute arbitrary code. Prefer "
        "safetensors / ONNX / GGUF, or load with torch.load(weights_only=True).",
        cwe=("CWE-502",),
        **_ARTIFACT,
    ),
    # --- containers ----------------------------------------------------------------------
    Rule(
        "LS110",
        "Archive structure anomaly",
        "ZIP container anomalies (duplicate member names, local/central header name mismatch, "
        "encryption flag, absolute or ../ member paths) that make scanners and loaders read "
        "different bytes.",
        cwe=("CWE-436", "CWE-22"),
        **_ARTIFACT,
    ),
    Rule(
        "LS111",
        "Unexpected member inside model archive",
        "The model archive contains an executable, script, or an additional pickle member.",
        cwe=("CWE-506",),
        **_ARTIFACT,
    ),
    Rule(
        "LS112",
        "TorchScript code inside archive",
        "TorchScript archives carry Python-like code executed by the TorchScript interpreter "
        "on torch.jit.load.",
        cwe=("CWE-94",),
        **_ARTIFACT,
    ),
    # --- tensor formats ------------------------------------------------------------------
    Rule(
        "LS120",
        "Invalid safetensors header",
        "Header length, JSON, dtypes, shapes or data offsets are inconsistent. Malformed "
        "headers target parser bugs and resource exhaustion.",
        cwe=("CWE-20",),
        **_STRUCT,
    ),
    Rule(
        "LS121",
        "Suspicious safetensors metadata",
        "__metadata__ contains non-string values, very long values, URLs or code-like text.",
        cwe=("CWE-20",),
        **_STRUCT,
    ),
    Rule(
        "LS130",
        "ONNX external data path traversal",
        "A tensor's external_data location is absolute or escapes the model directory, which "
        "lets a model read arbitrary files at load time.",
        cwe=("CWE-22",),
        **_ARTIFACT,
    ),
    Rule(
        "LS131",
        "ONNX custom-domain operator",
        "Nodes use an operator domain outside the standard ONNX / ONNX Runtime domains; "
        "running them requires loading a custom-op shared library.",
        cwe=("CWE-829",),
        **_ARTIFACT,
    ),
    Rule(
        "LS132",
        "Malformed ONNX protobuf",
        "The ONNX protobuf could not be parsed.",
        cwe=("CWE-20",),
        **_STRUCT,
    ),
    Rule(
        "LS140",
        "Malformed GGUF header",
        "GGUF magic, version, counts, string lengths or tensor dims are out of bounds; "
        "several GGML loader CVEs exploit exactly these fields.",
        cwe=("CWE-20", "CWE-190"),
        **_STRUCT,
    ),
    Rule(
        "LS141",
        "Chat template with template-injection patterns",
        "A Jinja chat template (GGUF tokenizer.chat_template or tokenizer_config.json) uses "
        "dunder attribute access or imports; such templates achieved RCE in loaders that "
        "rendered them without a sandbox (e.g. CVE-2024-34359).",
        cwe=("CWE-1336",),
        **_PAYLOAD,
    ),
    Rule(
        "LS150",
        "Keras Lambda layer",
        "A Keras Lambda layer stores a Python function (often marshalled bytecode) that runs "
        "when the model is deserialized or called (cf. CVE-2024-3660).",
        cwe=("CWE-94",),
        **_PAYLOAD,
    ),
    Rule(
        "LS151",
        "Keras config references a non-Keras module",
        "A layer/object config points at a module outside keras/tensorflow; with unsafe "
        "deserialization this imports arbitrary code (cf. CVE-2025-1550).",
        cwe=("CWE-94",),
        **_PAYLOAD,
    ),
    Rule(
        "LS160",
        "File extension does not match content",
        "The bytes do not match the format the extension promises (e.g. a pickle named "
        ".safetensors). Loaders that sniff content may deserialize it unsafely.",
        cwe=("CWE-434",),
        **_ARTIFACT,
    ),
    # --- repo / config -------------------------------------------------------------------
    Rule(
        "LS200",
        "Model requires remote code (trust_remote_code)",
        "config.json / tokenizer_config.json declares auto_map or custom pipelines, so "
        "loading with trust_remote_code=True executes Python shipped in the repo.",
        cwe=("CWE-829", "CWE-94"),
        **_CODE,
    ),
    Rule(
        "LS201",
        "Dangerous call in repository Python code",
        "Custom modeling/tokenizer code calls exec/eval/os.system/subprocess/network APIs.",
        cwe=("CWE-94", "CWE-78"),
        **_CODE,
    ),
    Rule(
        "LS202",
        "Executable or script file in model repository",
        "Native executables, shared libraries or shell scripts are not expected in a model "
        "repository.",
        cwe=("CWE-506",),
        **_ARTIFACT,
    ),
    Rule(
        "LS203",
        "No license declared",
        "No LICENSE file and no license in the model card front matter; provenance and "
        "usage rights are unknown.",
        owasp_llm=(SUPPLY,),
    ),
    Rule(
        "LS204",
        "Symlink escapes repository",
        "A symlink points outside the scanned directory.",
        cwe=("CWE-59",),
        **_STRUCT,
    ),
    Rule(
        "LS205",
        "Suspicious config/tokenizer field",
        "A config or tokenizer field contains URLs, long encoded blobs, code-like text or a "
        "cross-repository code reference.",
        cwe=("CWE-829",),
        **_STRUCT,
    ),
    Rule(
        "LS210",
        "Model download not pinned to a commit",
        "from_pretrained / hf_hub_download / snapshot_download without revision=<commit "
        "sha>: the upstream repo can change under you.",
        cwe=("CWE-829", "CWE-494"),
        **_STRUCT,
    ),
    Rule(
        "LS211",
        "trust_remote_code=True in source code",
        "Enabling trust_remote_code executes model-repo Python on load.",
        cwe=("CWE-829", "CWE-94"),
        **_CODE,
    ),
    Rule(
        "LS212",
        "Unsafe deserializer call in source code",
        "torch.load without weights_only=True, pickle/joblib/dill load, or "
        "np.load(allow_pickle=True).",
        cwe=("CWE-502",),
        **_ARTIFACT,
    ),
    Rule(
        "LS213",
        "Pickle weights without a safetensors alternative",
        "The repo ships pickle-based weights only.",
        cwe=("CWE-502",),
        **_ARTIFACT,
    ),
    Rule(
        "LS299",
        "Artifact could not be scanned",
        "The scanner hit an error or a limit; the artifact is unverified.",
        owasp_llm=(SUPPLY,),
    ),
]

RULES: dict[str, Rule] = {r.id: r for r in _RULES}


def get_rule(rule_id: str) -> Rule:
    return RULES[rule_id]
