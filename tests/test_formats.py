"""Format-specific scanners on crafted inputs."""

from __future__ import annotations

import io
import struct
import zipfile

from loadsafe.findings import Severity
from loadsafe.policy import Policy
from loadsafe.scanners.common import Ctx
from loadsafe.scanners.gguf_scan import scan_gguf
from loadsafe.scanners.onnx_scan import scan_onnx
from loadsafe.scanners.tensors import scan_safetensors
from loadsafe.scanners.zip_scan import scan_zip


def test_safetensors_duplicate_key():
    # Duplicate JSON keys are rejected by object_pairs_hook.
    header = (
        b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
        b'"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    )
    data = struct.pack("<Q", len(header)) + header + b"\x00\x00\x00\x00"
    ctx = Ctx("x", Policy())
    scan_safetensors(io.BytesIO(data), len(data), ctx)
    assert any(f.rule_id == "LS120" for f in ctx.findings)


def test_gguf_bad_magic():
    ctx = Ctx("x", Policy())
    scan_gguf(io.BytesIO(b"XXXX"), 4, ctx)
    assert any(f.rule_id == "LS140" and f.severity >= Severity.HIGH for f in ctx.findings)


def test_onnx_external_path_traversal():
    # Minimal ModelProto: ir_version=9 (field 1 varint), graph with one initializer
    # TensorProto with external_data location="../etc/passwd".
    # Hand-built protobuf is error-prone; use a tiny valid model via onnx if available,
    # otherwise skip.
    __import__("pytest").importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    # External tensor as an initializer
    t = helper.make_tensor("w", TensorProto.FLOAT, [1], [1.0])
    t.ClearField("float_data")
    t.data_location = TensorProto.EXTERNAL
    entry = t.external_data.add()
    entry.key = "location"
    entry.value = "../etc/passwd"
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y], [t])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    ctx = Ctx("x", Policy())
    scan_onnx(model.SerializeToString(), ctx)
    assert any(f.rule_id == "LS130" for f in ctx.findings)


def test_zip_path_traversal():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.pkl", b"\x80\x04.")
        z.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
    buf.seek(0)
    ctx = Ctx("x", Policy())
    scan_zip(buf, ctx)
    assert any(f.rule_id == "LS110" for f in ctx.findings)
