"""Static analysis of pickle streams without executing them.

A tiny symbolic pickle machine walks the opcode stream (decoded with the argument readers
from :mod:`pickletools`) and tracks the stack, metastack and memo symbolically. That lets us

* resolve STACK_GLOBAL operands even when they come from the memo,
* attribute every REDUCE/INST/OBJ/NEWOBJ to the callable it actually invokes (with arguments),
* spot calls on non-constant callables (getattr chains, call results),
* recurse into nested pickles passed as bytes constants,
* follow joblib's format, which interleaves raw NumPy buffers with pickle opcodes, and
* keep reporting findings that precede a corrupt/truncated tail (pickle executes as it reads).
"""

from __future__ import annotations

import io
import pickletools
from dataclasses import dataclass, field
from typing import IO, Any

from .findings import Finding, Severity
from .policy import Policy, Verdict

_CODE2OP: dict[str, pickletools.OpcodeInfo] = {op.code: op for op in pickletools.opcodes}
MAX_OPCODES = 50_000_000
MAX_NESTED_DEPTH = 3
_DTYPE_KIND_SIZES = {"U": 4}


class PickleParseError(Exception):
    pass


@dataclass(eq=False)
class Glob:
    module: str
    name: str
    verdict: Verdict

    @property
    def full(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass(eq=False)
class Call:
    func: Any
    args: Any
    kind: str
    state: Any = None


@dataclass(eq=False)
class Unknown:
    desc: str


@dataclass(eq=False)
class Persist:
    pid: Any


@dataclass(eq=False)
class PDict:
    pairs: list[tuple[Any, Any]] = field(default_factory=list)

    def get(self, key: str) -> Any:
        for k, v in self.pairs:
            if k == key:
                return v
        return None

    def has(self, key: str) -> bool:
        return any(k == key for k, _ in self.pairs)


@dataclass(eq=False)
class PSet:
    items: list[Any] = field(default_factory=list)
    frozen: bool = False


def safe_repr(value: Any, limit: int = 160, _depth: int = 0) -> str:
    """Bounded, side-effect free rendering of symbolic values."""
    if _depth > 4:
        return "…"
    if isinstance(value, Glob):
        out = value.full
    elif isinstance(value, Call):
        out = f"{safe_repr(value.func, limit, _depth + 1)}{safe_repr(value.args, limit, _depth + 1)}"
        if not isinstance(value.args, tuple):
            out = f"{safe_repr(value.func, limit, _depth + 1)}(*{safe_repr(value.args, limit, _depth + 1)})"
    elif isinstance(value, Unknown):
        out = f"<{value.desc}>"
    elif isinstance(value, Persist):
        out = f"<persistent {safe_repr(value.pid, limit, _depth + 1)}>"
    elif isinstance(value, PDict):
        inner = ", ".join(f"{safe_repr(k, limit, _depth + 1)}: {safe_repr(v, limit, _depth + 1)}"
                          for k, v in value.pairs[:8])
        out = "{" + inner + (", …" if len(value.pairs) > 8 else "") + "}"
    elif isinstance(value, PSet):
        out = "{" + ", ".join(safe_repr(v, limit, _depth + 1) for v in value.items[:8]) + "}"
    elif isinstance(value, tuple):
        inner = ", ".join(safe_repr(v, limit, _depth + 1) for v in value[:8])
        out = "(" + inner + ("," if len(value) == 1 else "") + (", …" if len(value) > 8 else "") + ")"
    elif isinstance(value, list):
        out = "[" + ", ".join(safe_repr(v, limit, _depth + 1) for v in value[:8]) + (
            ", …" if len(value) > 8 else "") + "]"
    elif isinstance(value, (bytes, bytearray)):
        out = repr(bytes(value[:48])) + (f"…({len(value)} bytes)" if len(value) > 48 else "")
    else:
        out = repr(value)
    return out if len(out) <= limit else out[: limit - 1] + "…"


@dataclass
class _ImportRecord:
    glob: Glob
    count: int = 0
    first_offset: int = 0
    location: str = ""
    calls: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class PickleScanResult:
    findings: list[Finding] = field(default_factory=list)
    globals: dict[str, str] = field(default_factory=dict)  # full name -> category
    opcodes: int = 0
    pickles: int = 0
    complete: bool = True
    bytes_consumed: int = 0


class _Machine:
    def __init__(self, f: IO[bytes], path: str, location: str, policy: Policy, depth: int,
                 result: PickleScanResult, imports: dict[str, _ImportRecord]) -> None:
        self.f = f
        self.path = path
        self.location = location
        self.policy = policy
        self.depth = depth
        self.result = result
        self.imports = imports
        self.stack: list[Any] = []
        self.metastack: list[list[Any]] = []
        self.memo: dict[int, Any] = {}

    # -- helpers ---------------------------------------------------------------------------
    def _loc(self, offset: int) -> str:
        return f"{self.location}@{offset}" if self.location else f"offset {offset}"

    def _add(self, rule: str, sev: Severity, msg: str, offset: int, **details: Any) -> None:
        self.result.findings.append(Finding(rule, sev, msg, self.path, self._loc(offset), details))

    def _pop(self) -> Any:
        if not self.stack:
            raise PickleParseError("stack underflow")
        return self.stack.pop()

    def _pop_mark(self) -> list[Any]:
        if not self.metastack:
            raise PickleParseError("MARK expected")
        items = self.stack
        self.stack = self.metastack.pop()
        return items

    def _glob(self, module: str, name: str, offset: int) -> Glob:
        verdict = self.policy.classify(module, name)
        g = Glob(module, name, verdict)
        rec = self.imports.get(g.full)
        if rec is None:
            rec = self.imports[g.full] = _ImportRecord(g, 0, offset, self._loc(offset))
        rec.count += 1
        self.result.globals[g.full] = verdict.category
        return g

    def _call(self, func: Any, args: Any, kind: str, offset: int) -> Call:
        call = Call(func, args, kind)
        if isinstance(func, Glob):
            rec = self.imports.get(func.full)
            if rec is not None and func.verdict.category == "dangerous" and len(rec.calls) < 5:
                rec.calls.append((offset, safe_repr(call, 300)))
        else:
            self._add("LS103", Severity.HIGH,
                      f"{kind} invokes a non-constant callable {safe_repr(func, 120)}", offset,
                      call=safe_repr(call, 300))
        self._scan_nested(args, offset)
        return call

    def _scan_nested(self, value: Any, offset: int, _depth: int = 0) -> None:
        if _depth > 3 or self.depth >= MAX_NESTED_DEPTH:
            return
        items: list[Any]
        if isinstance(value, (bytes, bytearray)):
            data = bytes(value)
            if len(data) >= 4 and data[0] == 0x80 and data[1] <= 5 and data[-1:] == b".":
                sub = scan_pickle(data, self.path, f"{self._loc(offset)}/nested", self.policy,
                                  depth=self.depth + 1, multi=False)
                if sub.complete and sub.bytes_consumed == len(data):
                    if any(f.severity >= Severity.MEDIUM for f in sub.findings):
                        self._add("LS105", Severity.HIGH,
                                  "bytes constant is a nested pickle with findings", offset,
                                  nested_findings=len(sub.findings))
                    self.result.findings.extend(sub.findings)
            return
        if isinstance(value, (tuple, list)):
            items = list(value)
        elif isinstance(value, PDict):
            items = [v for _, v in value.pairs]
        else:
            return
        for v in items[:64]:
            self._scan_nested(v, offset, _depth + 1)

    # -- joblib support ----------------------------------------------------------------------
    def _joblib_after_build(self, obj: Any, state: Any, offset: int) -> None:
        if not (isinstance(obj, Call) and isinstance(obj.func, Glob)
                and obj.func.module.startswith("joblib.numpy_pickle")
                and obj.func.name == "NumpyArrayWrapper" and isinstance(state, PDict)):
            return
        dtype = state.get("dtype")
        shape = state.get("shape")
        if not (isinstance(dtype, Call) and isinstance(dtype.args, tuple) and dtype.args
                and isinstance(dtype.args[0], str) and isinstance(shape, tuple)
                and all(isinstance(x, int) and x >= 0 for x in shape)):
            raise PickleParseError("unrecognised joblib NumpyArrayWrapper state")
        kind_str = dtype.args[0]
        if kind_str.startswith("O"):
            # object arrays are written as a nested protocol-5 pickle stream
            sub = _Machine(self.f, self.path, f"{self.location}/joblib-object-array",
                           self.policy, self.depth, self.result, self.imports)
            sub.run()
            return
        itemsize = _dtype_itemsize(kind_str, dtype.state)
        count = 1
        for x in shape:
            count *= x
        if state.has("numpy_array_alignment_bytes") and state.get(
                "numpy_array_alignment_bytes") is not None:
            pad = self.f.read(1)
            if len(pad) != 1:
                raise PickleParseError("truncated joblib padding")
            _skip(self.f, pad[0])
        _skip(self.f, count * itemsize)

    # -- main loop ---------------------------------------------------------------------------
    def run(self) -> None:
        f = self.f
        res = self.result
        st = self.stack
        while True:
            offset = f.tell()
            code = f.read(1)
            if not code:
                raise PickleParseError("unexpected end of stream (no STOP)")
            op = _CODE2OP.get(code.decode("latin-1"))
            if op is None:
                raise PickleParseError(f"invalid opcode {code!r} at {offset}")
            try:
                arg = op.arg.reader(f) if op.arg is not None else None
            except Exception as exc:  # pickletools raises ValueError on truncation
                raise PickleParseError(f"bad argument for {op.name}: {exc}") from exc
            res.opcodes += 1
            if res.opcodes > MAX_OPCODES:
                raise PickleParseError("opcode limit exceeded")
            name = op.name
            st = self.stack
            if name == "STOP":
                return
            if name in _PUSH_ARG:
                st.append(arg)
            elif name == "NONE":
                st.append(None)
            elif name == "NEWTRUE":
                st.append(True)
            elif name == "NEWFALSE":
                st.append(False)
            elif name in ("PROTO", "FRAME", "READONLY_BUFFER"):
                pass
            elif name == "EMPTY_LIST":
                st.append([])
            elif name == "EMPTY_TUPLE":
                st.append(())
            elif name == "EMPTY_DICT":
                st.append(PDict())
            elif name == "EMPTY_SET":
                st.append(PSet())
            elif name == "MARK":
                self.metastack.append(self.stack)
                self.stack = []
            elif name == "POP":
                if st:
                    st.pop()
                else:
                    self._pop_mark()
            elif name == "POP_MARK":
                self._pop_mark()
            elif name == "DUP":
                if not st:
                    raise PickleParseError("DUP on empty stack")
                st.append(st[-1])
            elif name == "APPEND":
                v = self._pop()
                if st and isinstance(st[-1], list):
                    st[-1].append(v)
                elif not st:
                    raise PickleParseError("APPEND on empty stack")
            elif name == "APPENDS":
                items = self._pop_mark()
                if not self.stack:
                    raise PickleParseError("APPENDS on empty stack")
                if isinstance(self.stack[-1], list):
                    self.stack[-1].extend(items)
                self._scan_list_for_nested(items, offset)
            elif name == "LIST":
                items = self._pop_mark()
                self.stack.append(items)
            elif name == "TUPLE":
                items = self._pop_mark()
                self.stack.append(tuple(items))
            elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                n = int(name[-1])
                if len(st) < n:
                    raise PickleParseError(f"{name} stack underflow")
                vals = tuple(st[-n:])
                del st[-n:]
                st.append(vals)
            elif name == "DICT":
                items = self._pop_mark()
                self.stack.append(PDict(list(zip(items[::2], items[1::2]))))
            elif name == "SETITEM":
                v = self._pop()
                k = self._pop()
                if not st:
                    raise PickleParseError("SETITEM on empty stack")
                if isinstance(st[-1], PDict):
                    st[-1].pairs.append((k, v))
                self._scan_nested(v, offset)
            elif name == "SETITEMS":
                items = self._pop_mark()
                if not self.stack:
                    raise PickleParseError("SETITEMS on empty stack")
                if isinstance(self.stack[-1], PDict):
                    self.stack[-1].pairs.extend(zip(items[::2], items[1::2]))
                self._scan_list_for_nested(items[1::2], offset)
            elif name == "ADDITEMS":
                items = self._pop_mark()
                if self.stack and isinstance(self.stack[-1], PSet):
                    self.stack[-1].items.extend(items)
            elif name == "FROZENSET":
                items = self._pop_mark()
                self.stack.append(PSet(items, frozen=True))
            elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
                if not st:
                    raise PickleParseError("PUT on empty stack")
                self.memo[int(arg)] = st[-1]
            elif name == "MEMOIZE":
                if not st:
                    raise PickleParseError("MEMOIZE on empty stack")
                self.memo[len(self.memo)] = st[-1]
            elif name in ("GET", "BINGET", "LONG_BINGET"):
                key = int(arg)
                if key not in self.memo:
                    raise PickleParseError(f"memo key {key} missing")
                st.append(self.memo[key])
            elif name == "GLOBAL":
                module, _, gname = str(arg).partition(" ")
                st.append(self._glob(module, gname, offset))
            elif name == "STACK_GLOBAL":
                gname = self._pop()
                module = self._pop()
                if isinstance(module, str) and isinstance(gname, str):
                    self.stack.append(self._glob(module, gname, offset))
                else:
                    self._add("LS103", Severity.HIGH, "STACK_GLOBAL with non-constant operands "
                              f"({safe_repr(module, 60)}, {safe_repr(gname, 60)})", offset)
                    self.stack.append(Unknown("dynamic global"))
            elif name == "INST":
                module, _, gname = str(arg).partition(" ")
                items = self._pop_mark()
                g = self._glob(module, gname, offset)
                self.stack.append(self._call(g, tuple(items), "INST", offset))
            elif name == "OBJ":
                items = self._pop_mark()
                if not items:
                    raise PickleParseError("OBJ without class")
                self.stack.append(self._call(items[0], tuple(items[1:]), "OBJ", offset))
            elif name == "REDUCE":
                args = self._pop()
                func = self._pop()
                self.stack.append(self._call(func, args, "REDUCE", offset))
            elif name == "NEWOBJ":
                args = self._pop()
                cls = self._pop()
                self.stack.append(self._call(cls, args, "NEWOBJ", offset))
            elif name == "NEWOBJ_EX":
                kwargs = self._pop()
                args = self._pop()
                cls = self._pop()
                self.stack.append(self._call(cls, (args, kwargs), "NEWOBJ_EX", offset))
            elif name == "BUILD":
                state = self._pop()
                if not self.stack:
                    raise PickleParseError("BUILD on empty stack")
                obj = self.stack[-1]
                if isinstance(obj, Call):
                    obj.state = state
                elif not isinstance(obj, (PDict, list, Glob, Unknown, Persist)):
                    pass
                self._scan_nested(state, offset)
                self._joblib_after_build(obj, state, offset)
            elif name in ("EXT1", "EXT2", "EXT4"):
                self._add("LS106", Severity.MEDIUM,
                          f"{name} resolves a global via the copyreg extension registry (code "
                          f"{arg})", offset)
                st.append(Unknown(f"extension {arg}"))
            elif name == "PERSID":
                st.append(Persist(arg))
            elif name == "BINPERSID":
                st.append(Persist(self._pop()))
            elif name == "NEXT_BUFFER":
                st.append(Unknown("out-of-band buffer"))
            else:  # pragma: no cover - every opcode in pickletools is handled above
                raise PickleParseError(f"unhandled opcode {name}")

    def _scan_list_for_nested(self, items: list[Any], offset: int) -> None:
        for v in items[:64]:
            self._scan_nested(v, offset)


_PUSH_ARG = frozenset({
    "INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "STRING", "BINSTRING",
    "SHORT_BINSTRING", "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8", "UNICODE",
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "FLOAT", "BINFLOAT",
})

def _dtype_itemsize(kind: str, state: Any) -> int:
    if isinstance(state, tuple) and len(state) >= 6 and isinstance(state[5], int) and state[5] > 0:
        return state[5]
    code = kind.lstrip("<>|=")
    letter = code[:1]
    digits = "".join(ch for ch in code[1:] if ch.isdigit())
    if letter == "?" or (letter == "b" and not digits):
        return 1
    if not digits:
        raise PickleParseError(f"cannot size dtype {kind!r}")
    return int(digits) * _DTYPE_KIND_SIZES.get(letter, 1)


def _skip(f: IO[bytes], n: int) -> None:
    if n <= 0:
        return
    if f.seekable():
        here = f.tell()
        f.seek(n, io.SEEK_CUR)
        if f.tell() != here + n or (hasattr(f, "getbuffer") and f.tell() > len(f.getbuffer())):
            raise PickleParseError("truncated array data")
        return
    remaining = n
    while remaining:
        chunk = f.read(min(remaining, 1 << 20))
        if not chunk:
            raise PickleParseError("truncated array data")
        remaining -= len(chunk)


def scan_pickle(data: bytes | IO[bytes], path: str, location: str = "",
                policy: Policy | None = None, depth: int = 0, multi: bool = True,
                size_hint: int | None = None) -> PickleScanResult:
    """Scan one pickle stream (or, with ``multi``, consecutive pickles) symbolically."""
    policy = policy or Policy()
    f: IO[bytes] = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
    result = PickleScanResult()
    imports: dict[str, _ImportRecord] = {}
    index = 0
    while True:
        start = f.tell()
        head = f.read(1)
        if not head:
            break
        f.seek(start)
        loc = location if index == 0 else f"{location}#pickle{index + 1}"
        machine = _Machine(f, path, loc, policy, depth, result, imports)
        n_before = len(result.findings)
        imports_before = set(imports)
        try:
            machine.run()
            result.pickles += 1
        except PickleParseError as exc:
            if index == 0:
                result.complete = False
                result.findings.append(Finding(
                    "LS104", Severity.MEDIUM, f"pickle stream could not be fully parsed: {exc}",
                    path, f"{location}@{f.tell()}" if location else f"offset {f.tell()}",
                    {"error": str(exc)}))
            else:
                # Trailing bytes after a complete pickle (e.g. raw storages in legacy torch
                # files). Only trust what a *complete* follow-up pickle would contain.
                del result.findings[n_before:]
                for k in set(imports) - imports_before:
                    del imports[k]
            f.seek(start)
            break
        index += 1
        if not multi:
            break
    result.bytes_consumed = f.tell()
    _finalize(result, imports, path)
    return result


def _finalize(result: PickleScanResult, imports: dict[str, _ImportRecord], path: str) -> None:
    for full, rec in imports.items():
        v = rec.glob.verdict
        loc = rec.location
        if v.category == "dangerous":
            if rec.calls:
                for _off, call in rec.calls:
                    result.findings.append(Finding(
                        "LS100", Severity.CRITICAL, f"unpickling calls {call}", path, loc,
                        {"global": full, "reason": v.reason, "call": call}))
            else:
                result.findings.append(Finding(
                    "LS101", Severity.HIGH, f"pickle imports dangerous global {full}", path, loc,
                    {"global": full, "reason": v.reason, "count": rec.count}))
        elif v.category in ("unknown", "ml-library"):
            result.findings.append(Finding(
                "LS102", v.severity, f"pickle imports non-allowlisted global {full}", path, loc,
                {"global": full, "reason": v.reason, "category": v.category,
                 "count": rec.count}))
