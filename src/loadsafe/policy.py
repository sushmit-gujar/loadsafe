"""Classification of pickle globals into safe / ML-library / unknown / dangerous.

Deny rules always win over allow rules. Unknown globals are reported as MEDIUM so that new
gadgets that are not in the denylist still surface (the main failure mode of denylist-only
scanners).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .findings import Severity

# Entire modules whose members can execute code, spawn processes, touch the network or
# filesystem, or deserialize further untrusted data.
DANGEROUS_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "posix",
        "nt",
        "posixpath",
        "ntpath",
        "subprocess",
        "_posixsubprocess",
        "sys",
        "socket",
        "_socket",
        "ssl",
        "shutil",
        "runpy",
        "importlib",
        "imp",
        "zipimport",
        "pkgutil",
        "ctypes",
        "_ctypes",
        "cffi",
        "pty",
        "commands",
        "popen2",
        "platform",
        "pickle",
        "_pickle",
        "cPickle",
        "pickletools",
        "dill",
        "cloudpickle",
        "marshal",
        "shelve",
        "code",
        "codeop",
        "pdb",
        "bdb",
        "cProfile",
        "profile",
        "trace",
        "timeit",
        "doctest",
        "webbrowser",
        "urllib",
        "urllib2",
        "urllib3",
        "http",
        "httplib",
        "requests",
        "httpx",
        "aiohttp",
        "ftplib",
        "smtplib",
        "telnetlib",
        "poplib",
        "imaplib",
        "nntplib",
        "xmlrpc",
        "socketserver",
        "asyncio",
        "multiprocessing",
        "concurrent",
        "threading",
        "_thread",
        "signal",
        "pydoc",
        "idlelib",
        "lib2to3",
        "pip",
        "setuptools",
        "distutils",
        "venv",
        "ensurepip",
        "site",
        "sysconfig",
        "tempfile",
        "glob",
        "fileinput",
        "io",
        "_io",
        "linecache",
        "tarfile",
        "zipfile",
        "gzip",
        "bz2",
        "lzma",
        "sqlite3",
        "dbm",
        "mmap",
        "fcntl",
        "resource",
        "gc",
        "inspect",
        "types",
        "weakref",
        "atexit",
        "operator",
        "_operator",
        "functools",
        "copyreg",
        "copy_reg",
        "uuid",
        "jinja2",
        "mako",
        "yaml",
        "jsonpickle",
        "IPython",
        "ipykernel",
        "torch.hub",
        "torch.utils.cpp_extension",
        "torch.utils.bottleneck",
        "torch._dynamo",
        "torch._inductor",
        "torch.jit",
        "torch.package",
        "torch.distributed",
        "torch.multiprocessing",
        "torch.serialization",
        "torch._C._jit",
        "torch.testing",
        "numpy.f2py",
        "numpy.distutils",
        "numpy.testing",
        "numpy.lib.npyio",
        "numpy.load",
        "pandas.io",
        "huggingface_hub",
        "transformers.dynamic_module_utils",
        "keras.utils",
        "tensorflow.io",
        "joblib.parallel",
        "sklearn.externals",
        "smtpd",
        "ast",
        "compileall",
        "py_compile",
        "tokenize",
    }
)

# Specific (module, name) pairs that are dangerous even if the module is otherwise allowed.
DANGEROUS_NAMES: frozenset[tuple[str, str]] = frozenset(
    {
        ("torch", "load"),
        ("torch", "save"),
        ("torch", "compile"),
        ("torch", "_load_global_deps"),
        ("torch.storage", "_load_from_bytes"),
        ("torch._utils", "_load_global_deps"),
        ("numpy", "load"),
        ("numpy", "loadtxt"),
        ("numpy", "fromfile"),
        ("numpy", "save"),
        ("numpy.lib.format", "read_array"),
        ("numpy.core.multiarray", "fromfile"),
        ("pandas", "read_pickle"),
        ("pandas", "read_csv"),
        ("pandas", "eval"),
        ("pandas.core.computation.eval", "eval"),
        ("joblib", "load"),
        ("joblib", "dump"),
    }
)

DANGEROUS_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "execfile",
        "compile",
        "__import__",
        "open",
        "file",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "input",
        "raw_input",
        "help",
        "memoryview",
        "apply",
        "reload",
        "exit",
        "quit",
        "super",
    }
)

# Known-safe globals that appear in normal ML checkpoints.
SAFE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("collections", "deque"),
        ("collections", "Counter"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "tuple"),
        ("builtins", "slice"),
        ("builtins", "range"),
        ("builtins", "complex"),
        ("builtins", "bytearray"),
        ("builtins", "bytes"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "str"),
        ("builtins", "bool"),
        ("builtins", "object"),
        ("__builtin__", "object"),
        ("__builtin__", "set"),
        ("__builtin__", "frozenset"),
        ("copy_reg", "_reconstructor"),
        ("copyreg", "_reconstructor"),
        ("torch._utils", "_rebuild_tensor"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_tensor_v3"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_parameter_with_state"),
        ("torch._utils", "_rebuild_qtensor"),
        ("torch._utils", "_rebuild_sparse_tensor"),
        ("torch._utils", "_rebuild_nested_tensor"),
        ("torch._utils", "_rebuild_meta_tensor_no_storage"),
        ("torch._utils", "_rebuild_wrapper_subclass"),
        ("torch._utils", "_rebuild_device_tensor_from_numpy"),
        ("torch._tensor", "_rebuild_from_type_v2"),
        ("torch", "Size"),
        ("torch", "device"),
        ("torch", "dtype"),
        ("torch", "Tensor"),
        ("torch.nn.parameter", "Parameter"),
        ("torch", "per_tensor_affine"),
        ("torch", "per_channel_affine"),
        ("torch.serialization", "_get_layout"),
        ("torch", "strided"),
        ("torch", "sparse_coo"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy", "core.multiarray._reconstruct"),
        ("numpy.random._pickle", "__randomstate_ctor"),
        ("numpy.random._pickle", "__bit_generator_ctor"),
        ("numpy.random._pickle", "__generator_ctor"),
        ("numpy.random", "__RandomState_ctor"),
        ("numpy.ma.core", "_mareconstruct"),
        ("joblib.numpy_pickle", "NumpyArrayWrapper"),
        ("joblib.numpy_pickle_compat", "NDArrayWrapper"),
        ("joblib.numpy_pickle_compat", "ZNDArrayWrapper"),
        ("datetime", "datetime"),
        ("datetime", "date"),
        ("datetime", "time"),
        ("datetime", "timedelta"),
        ("datetime", "timezone"),
        ("decimal", "Decimal"),
        ("fractions", "Fraction"),
        ("pathlib", "PosixPath"),
        ("pathlib", "WindowsPath"),
        ("pathlib", "PurePosixPath"),
        ("argparse", "Namespace"),
        ("_codecs", "encode"),
        ("codecs", "encode"),
        ("re", "_compile"),
        ("enum", "Enum"),
    }
)

SAFE_TORCH_STORAGE_SUFFIX = "Storage"  # torch.FloatStorage, torch.HalfStorage, ...

# Namespaces of mainstream ML libraries whose classes routinely appear in pickles
# (scikit-learn estimators, pandas frames, ...). Reported as LOW so they are visible
# but do not fail a build by default.
ML_LIBRARY_PREFIXES: tuple[str, ...] = (
    "sklearn",
    "numpy",
    "scipy",
    "pandas",
    "torch",
    "xgboost",
    "lightgbm",
    "catboost",
    "transformers",
    "tokenizers",
    "sentence_transformers",
    "torchvision",
    "torchaudio",
    "timm",
    "statsmodels",
    "joblib",
    "tensorflow",
    "keras",
    "jax",
    "flax",
    "optax",
    "accelerate",
    "peft",
    "diffusers",
    "spacy",
    "thinc",
    "gensim",
    "nltk",
    "fasttext",
    "skops",
    "imblearn",
    "category_encoders",
    "pytorch_lightning",
    "lightning",
    "omegaconf",
    "tqdm",
    "pyarrow",
    "_codecs",
)


def _module_matches(module: str, prefixes: frozenset[str] | tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


@dataclass
class Verdict:
    severity: Severity
    category: str  # "safe" | "ml-library" | "unknown" | "dangerous"
    reason: str


@dataclass
class Policy:
    """Global-classification policy. Extra allow/deny entries use 'module.name' or 'module.*'."""

    extra_allow: set[str] = field(default_factory=set)
    extra_deny: set[str] = field(default_factory=set)

    @staticmethod
    def _matches(module: str, name: str, patterns: set[str]) -> bool:
        full = f"{module}.{name}"
        for p in patterns:
            if p.endswith(".*"):
                if _module_matches(module, (p[:-2],)):
                    return True
            elif p == full:
                return True
        return False

    def classify(self, module: str, name: str) -> Verdict:
        if "." in name:
            # Protocol 4+ resolves dotted names with getattr chains, e.g.
            # ("torch", "serialization.os.system"): check every hop.
            parts = name.split(".")
            tops = {m.split(".")[0] for m in DANGEROUS_MODULES}
            for part in parts:
                if part.startswith("__") and part.endswith("__"):
                    return Verdict(
                        Severity.HIGH,
                        "dangerous",
                        f"dotted name {module}.{name} traverses dunder {part}",
                    )
                if part in tops and part not in ("io", "types", "copy", "site"):
                    return Verdict(
                        Severity.HIGH,
                        "dangerous",
                        f"dotted name {module}.{name} reaches module {part}",
                    )
            return self._classify(module + "." + ".".join(parts[:-1]), parts[-1], module, name)
        return self._classify(module, name, module, name)

    def _classify(self, module: str, name: str, orig_mod: str, orig_name: str) -> Verdict:
        full = f"{module}.{name}"
        if self._matches(module, name, self.extra_deny) or self._matches(
            orig_mod, orig_name, self.extra_deny
        ):
            return Verdict(Severity.HIGH, "dangerous", f"{full} is in the user denylist")
        if module in ("builtins", "__builtin__") and name in DANGEROUS_BUILTINS:
            return Verdict(
                Severity.HIGH,
                "dangerous",
                f"builtin {name} enables code execution or attribute access",
            )
        if (module, name) in DANGEROUS_NAMES:
            return Verdict(Severity.HIGH, "dangerous", f"{full} loads/evaluates further data")
        if (module, name) in SAFE_GLOBALS or (orig_mod, orig_name) in SAFE_GLOBALS:
            return Verdict(Severity.INFO, "safe", "known-safe ML global")
        if module == "torch" and name.endswith(SAFE_TORCH_STORAGE_SUFFIX):
            return Verdict(Severity.INFO, "safe", "torch storage type")
        if _module_matches(module, DANGEROUS_MODULES):
            return Verdict(
                Severity.HIGH,
                "dangerous",
                f"module {module} can execute code, spawn processes or perform I/O",
            )
        if name.startswith("__") and name.endswith("__"):
            return Verdict(Severity.HIGH, "dangerous", f"dunder attribute {name} accessed")
        if self._matches(module, name, self.extra_allow) or self._matches(
            orig_mod, orig_name, self.extra_allow
        ):
            return Verdict(Severity.INFO, "safe", "in the user allowlist")
        if _module_matches(module, ML_LIBRARY_PREFIXES):
            return Verdict(Severity.LOW, "ml-library", f"class/function from ML library {module}")
        return Verdict(Severity.MEDIUM, "unknown", f"{full} is not in the allowlist")
