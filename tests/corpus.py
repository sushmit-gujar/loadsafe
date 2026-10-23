"""Generate a corpus of HARMLESS proof-of-concept malicious artifacts and benign ones.

Nothing malicious is committed to the repository: every artifact is built at run time in a
temp dir. The malicious PoCs' "payloads" are inert -- they write a marker file into a
temp dir or call ``builtins.print`` -- but they use exactly the pickle/zip/format
constructs that real attacks use, so the scanner is exercised honestly.
"""

from __future__ import annotations

import json
import os
import pickle
import struct
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

MARKER = Path(tempfile.gettempdir()) / "loadsafe_poc_marker.txt"


class _ReducePrint:
    def __reduce__(self):
        return (print, ("loadsafe-poc: inert print payload",))


class _ReduceWriteMarker:
    def __reduce__(self):
        return (os.system, (f"echo loadsafe-poc > '{MARKER}'",))


class _ReduceExec:
    def __reduce__(self):
        return (exec, ("marker = 1  # inert",))


class _ReduceEval:
    def __reduce__(self):
        return (eval, ("__import__('sys').version",))


class _ReduceGetattr:
    def __reduce__(self):
        return (getattr, (str, "__class__"))


@dataclass
class Case:
    name: str
    malicious: bool
    build: Callable[[Path], Path]
    note: str = ""
    expect_rules: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)


def _w(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _benign_state_dict(p: Path) -> Path:
    import numpy as np

    sd = {
        "linear.weight": np.zeros((4, 4), dtype=np.float32),
        "linear.bias": np.zeros(4, dtype=np.float32),
        "config": {"n": 4},
    }
    return _w(p / "state_dict.pkl", pickle.dumps(sd, protocol=2))


def _benign_sklearn(p: Path) -> Path:
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(max_iter=10).fit(
        np.random.RandomState(0).rand(30, 4), np.arange(30) % 2
    )
    out = p / "model.joblib"
    joblib.dump(m, out)
    return out


def _benign_torch(p: Path) -> Path:
    import torch

    out = p / "weights.pt"
    torch.save({"w": torch.zeros(8, 8), "meta": {"epoch": 3}}, out)
    return out


def _benign_npy(p: Path) -> Path:
    import numpy as np

    out = p / "array.npy"
    np.save(out, np.arange(100, dtype=np.float64).reshape(10, 10))
    return out


def _benign_safetensors(p: Path) -> Path:
    import numpy as np
    from safetensors.numpy import save_file

    out = p / "model.safetensors"
    save_file(
        {"w": np.zeros((16, 16), dtype=np.float32), "b": np.zeros(16, dtype=np.float32)},
        str(out),
        metadata={"framework": "pt", "note": "benign"},
    )
    return out


def _benign_gguf(p: Path) -> Path:
    import gguf
    import numpy as np

    out = p / "tiny.gguf"
    w = gguf.GGUFWriter(str(out), "llama")
    w.add_uint32("llama.context_length", 128)
    w.add_string("general.name", "tiny-benign")
    w.add_tensor("tok.weight", np.zeros((8, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out


def _benign_onnx(p: Path) -> Path:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    w = helper.make_tensor("w", TensorProto.FLOAT, [4], np.ones(4, dtype=np.float32))
    node = helper.make_node("Mul", ["x", "w"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    out = p / "model.onnx"
    onnx.save(model, str(out))
    return out


def _benign_keras(p: Path) -> Path:
    cfg = {
        "class_name": "Sequential",
        "config": {
            "name": "seq",
            "layers": [
                {"class_name": "Dense", "config": {"name": "d1", "units": 8}},
                {"class_name": "Dense", "config": {"name": "d2", "units": 1}},
            ],
        },
    }
    out = p / "benign.keras"
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("config.json", json.dumps(cfg))
        z.writestr("metadata.json", json.dumps({"keras_version": "3.0"}))
    return out


def _mal_pickle(obj: object, proto: int) -> Callable[[Path], Path]:
    def build(p: Path) -> Path:
        return _w(p / f"evil_p{proto}.pkl", pickle.dumps(obj, protocol=proto))

    return build


def _mal_torch_zip(p: Path) -> Path:
    data = pickle.dumps(_ReduceWriteMarker(), protocol=2)
    out = p / "evil_model.pt"
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("archive/data.pkl", data)
        z.writestr("archive/version", "3")
    return out


def _mal_npy_object(p: Path) -> Path:
    import numpy as np

    out = p / "evil_array.npy"
    np.save(out, np.array([_ReducePrint()], dtype=object), allow_pickle=True)
    return out


def _mal_pickle_as_safetensors(p: Path) -> Path:
    return _w(p / "model.safetensors", pickle.dumps(_ReduceWriteMarker(), protocol=2))


def _mal_nested_pickle(p: Path) -> Path:
    inner = pickle.dumps(_ReduceEval(), protocol=4)
    return _w(p / "nested.pkl", pickle.dumps({"payload": inner, "shape": [1]}, protocol=4))


def _mal_safetensors_bad_header(p: Path) -> Path:
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [10, 10], "data_offsets": [0, 999999]}}
    ).encode()
    return _w(p / "corrupt.safetensors", struct.pack("<Q", len(header)) + header + b"\x00" * 16)


def _mal_safetensors_code_metadata(p: Path) -> Path:
    import numpy as np
    from safetensors.numpy import save_file

    out = p / "meta.safetensors"
    save_file(
        {"w": np.zeros(4, dtype=np.float32)},
        str(out),
        metadata={"loader": "__import__('os').system('id')"},
    )
    return out


def _mal_gguf_ssti(p: Path) -> Path:
    import gguf
    import numpy as np

    out = p / "evil.gguf"
    w = gguf.GGUFWriter(str(out), "llama")
    w.add_string(
        "tokenizer.chat_template",
        "{{ self.__init__.__globals__.__builtins__.exec('x=1') }}",
    )
    w.add_tensor("t", np.zeros((8, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out


def _mal_gguf_bad(p: Path) -> Path:
    return _w(p / "bad.gguf", b"GGUF" + struct.pack("<IQQ", 3, 1 << 40, 1 << 40))


def _mal_onnx_pythonop(p: Path) -> Path:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("PythonOp", ["x"], ["y"], domain="com.evil")
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.evil", 1)],
    )
    model.ir_version = 9
    out = p / "pythonop.onnx"
    with open(out, "wb") as f:
        f.write(model.SerializeToString())
    return out


def _mal_keras_lambda(p: Path) -> Path:
    cfg = {
        "class_name": "Sequential",
        "config": {
            "name": "s",
            "layers": [
                {
                    "class_name": "Lambda",
                    "config": {
                        "name": "lam",
                        "function": {
                            "class_name": "__lambda__",
                            "config": {"code": "4wEAAAA=", "defaults": None},
                        },
                    },
                }
            ],
        },
    }
    out = p / "evil.keras"
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("config.json", json.dumps(cfg))
        z.writestr("metadata.json", json.dumps({"keras_version": "3.0"}))
    return out


def benign_cases() -> list[Case]:
    return [
        Case("benign_pickle_state_dict", False, _benign_state_dict),
        Case("benign_sklearn_joblib", False, _benign_sklearn),
        Case("benign_torch_pt", False, _benign_torch),
        Case("benign_npy", False, _benign_npy),
        Case("benign_safetensors", False, _benign_safetensors),
        Case("benign_gguf", False, _benign_gguf),
        Case("benign_onnx", False, _benign_onnx),
        Case("benign_keras", False, _benign_keras),
    ]


def malicious_cases() -> list[Case]:
    cases = [
        Case(
            f"mal_pickle_print_p{pr}",
            True,
            _mal_pickle(_ReducePrint(), pr),
            "REDUCE builtins.print",
            ("LS100",),
        )
        for pr in (0, 2, 4, 5)
    ]
    cases += [
        Case(
            "mal_pickle_os_system",
            True,
            _mal_pickle(_ReduceWriteMarker(), 2),
            "REDUCE os.system",
            ("LS100",),
        ),
        Case(
            "mal_pickle_exec",
            True,
            _mal_pickle(_ReduceExec(), 4),
            "REDUCE exec",
            ("LS100",),
        ),
        Case(
            "mal_pickle_eval",
            True,
            _mal_pickle(_ReduceEval(), 4),
            "REDUCE eval",
            ("LS100",),
        ),
        Case(
            "mal_pickle_getattr",
            True,
            _mal_pickle(_ReduceGetattr(), 4),
            "REDUCE getattr",
            ("LS100",),
        ),
        Case(
            "mal_torch_zip",
            True,
            _mal_torch_zip,
            "zip .pt with os.system",
            ("LS100",),
        ),
        Case(
            "mal_npy_object_pickle",
            True,
            _mal_npy_object,
            "object .npy with print",
            ("LS100", "LS107"),
        ),
        Case(
            "mal_pickle_as_safetensors",
            True,
            _mal_pickle_as_safetensors,
            "pickle named .safetensors",
            ("LS160",),
        ),
        Case(
            "mal_nested_pickle",
            True,
            _mal_nested_pickle,
            "pickle-in-pickle",
            ("LS105",),
        ),
        Case(
            "mal_safetensors_bad_header",
            True,
            _mal_safetensors_bad_header,
            "offsets past EOF",
            ("LS120",),
        ),
        Case(
            "mal_safetensors_code_meta",
            True,
            _mal_safetensors_code_metadata,
            "code-like metadata",
            ("LS121",),
        ),
        Case("mal_gguf_ssti", True, _mal_gguf_ssti, "GGUF SSTI", ("LS141",)),
        Case("mal_gguf_bad_counts", True, _mal_gguf_bad, "GGUF bad counts", ("LS140",)),
        Case("mal_onnx_pythonop", True, _mal_onnx_pythonop, "ONNX PythonOp", ("LS131",)),
        Case("mal_keras_lambda", True, _mal_keras_lambda, "Keras Lambda", ("LS150",)),
    ]
    return cases


def all_cases() -> list[Case]:
    return malicious_cases() + benign_cases()


def materialize(cases: list[Case], root: Path) -> list[tuple[Case, Path]]:
    out = []
    for c in cases:
        d = root / c.name
        d.mkdir(parents=True, exist_ok=True)
        out.append((c, c.build(d)))
    return out
