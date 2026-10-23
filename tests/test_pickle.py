"""Unit tests for the symbolic pickle machine."""

from __future__ import annotations

import pickle

from loadsafe.findings import Severity
from loadsafe.pickle_scan import scan_pickle
from loadsafe.policy import Policy


def test_empty_dict_all_protocols():
    for proto in range(6):
        r = scan_pickle(pickle.dumps({"a": [1, 2, 3]}, protocol=proto), "x")
        assert r.complete
        assert not any(f.severity >= Severity.MEDIUM for f in r.findings)


def test_ordereddict_is_safe():
    from collections import OrderedDict

    r = scan_pickle(pickle.dumps(OrderedDict(a=1)), "x")
    assert r.globals.get("collections.OrderedDict") == "safe"
    assert not any(f.severity >= Severity.MEDIUM for f in r.findings)


def test_allowlist_override():
    # Allowlist must not override the denylist (eval stays dangerous).
    class X:
        def __reduce__(self):
            return (eval, ("1+1",))

    r = scan_pickle(pickle.dumps(X(), protocol=4), "x",
                    policy=Policy(extra_allow={"builtins.eval"}))
    assert any(f.rule_id == "LS100" for f in r.findings)

    # Allowlist does silence unknown (medium) globals.
    class Y:
        def __reduce__(self):
            import decimal
            return (decimal.Decimal, ("1",))

    r = scan_pickle(pickle.dumps(Y(), protocol=4), "x")
    # Decimal is already safe; craft an unknown via a custom module name using pickletools.
    # Use --allow for an ML-library class treated as LOW.
    from collections import UserDict
    r = scan_pickle(pickle.dumps(UserDict(a=1), protocol=4), "x",
                    policy=Policy(extra_allow={"collections.UserDict"}))
    assert not any(f.rule_id == "LS102" and "UserDict" in f.message for f in r.findings)


def test_denylist_override():
    from collections import OrderedDict

    r = scan_pickle(
        pickle.dumps(OrderedDict(a=1)),
        "x",
        policy=Policy(extra_deny={"collections.OrderedDict"}),
    )
    assert any(f.severity >= Severity.HIGH for f in r.findings)


def test_stack_global_dynamic():
    # Craft a STREAM where STACK_GLOBAL operands are non-constant using BINGET tricks
    # is hard; instead verify our UNKNOWN path via an EXT opcode.
    stream = b"\x80\x04\x82\x01."  # PROTO 4, EXT1 1, STOP
    r = scan_pickle(stream, "x")
    assert any(f.rule_id == "LS106" for f in r.findings)


def test_truncated_still_flags_prior_payload():
    class X:
        def __reduce__(self):
            return (eval, ("1",))

    data = pickle.dumps(X(), protocol=4) + b"\xff\xff"
    r = scan_pickle(data, "x")
    # First pickle completes; trailing garbage is ignored under multi=True
    assert any(f.rule_id == "LS100" for f in r.findings)


def test_nested_pickle_bytes():
    class X:
        def __reduce__(self):
            return (eval, ("1",))

    inner = pickle.dumps(X(), protocol=4)
    outer = pickle.dumps({"payload": inner}, protocol=4)
    r = scan_pickle(outer, "x")
    assert any(f.rule_id == "LS105" for f in r.findings)
    assert any(f.rule_id == "LS100" for f in r.findings)
