#!/usr/bin/env python3
"""Audit numeric claims in experiment-result files against hard information-theory bounds.

Designed to be pointed at a directory of ``exp*.json`` / ``*_verify.json`` /
``*_check.json`` artifacts (and their ``.log`` companions) and to answer one
question per claim: *is this number physically/information-theoretically
possible?*

Stdlib only -- no numpy, no third-party deps -- so it runs anywhere Python 3.8+
runs, including the environment that produced the results.

Usage
-----
    python3 tools/audit_claims.py PATH [PATH ...]
    python3 tools/audit_claims.py --json results.json   # machine readable

Exit codes
----------
    0  no hard-bound violation found
    1  at least one claim violates a hard bound
    2  nothing could be audited (missing input)
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import re
import struct
import sys
import zlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# claim discovery
# ---------------------------------------------------------------------------

CLAIM_KEY = re.compile(
    r"(ratio|compress|snr|psnr|entropy|bits?_per|bps|bpp|flops|speedup|"
    r"speed_up|accuracy|error|lossless|roundtrip|round_trip|floor|bound|"
    r"theory|theoretical|achieved|actual|whiten|distortion|mse|rmse|drift|"
    r"max_?abs|recovery|fidelity|score|gain|throughput|hz|frequency|sampl)",
    re.IGNORECASE,
)

# keys whose value is asserted to be a *lossless* property of a codec
LOSSLESS_KEY = re.compile(r"(lossless|roundtrip|round_trip|exact|bitexact|bit_exact)", re.IGNORECASE)

RATIO_KEY = re.compile(r"(ratio|compress|compressing)", re.IGNORECASE)
DB_KEY = re.compile(r"(psnr|snr|db|decibel)", re.IGNORECASE)

JSON_SUFFIXES = (".json",)
LOG_SUFFIXES = (".log", ".txt")

MAX_ARRAY_AUDIT = 1_000_000  # cap on samples we push through a compressor


class Verdict:
    OK = "OK"
    IMPOSSIBLE = "IMPOSSIBLE"
    SUSPECT = "SUSPECT"
    UNVERIFIABLE = "UNVERIFIABLE"


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _flat_nums(node: Any, prefix: str, out: List[Tuple[str, float]]) -> None:
    """Collect every numeric leaf with its dotted path."""
    if _is_num(node):
        out.append((prefix or "<root>", float(node)))
    elif isinstance(node, dict):
        for k, v in node.items():
            _flat_nums(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if _is_num(v):
                out.append((f"{prefix}[]", float(v)))
            else:
                _flat_nums(v, f"{prefix}[{i}]", out)


def _num_arrays(node: Any, prefix: str, out: List[Tuple[str, Sequence[float]]]) -> None:
    """Collect every homogeneous numeric array of meaningful length."""
    if isinstance(node, dict):
        for k, v in node.items():
            _num_arrays(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(node, list):
        if len(node) >= 16 and all(_is_num(v) for v in node):
            out.append((prefix or "<root>", [float(v) for v in node]))
        else:
            for i, v in enumerate(node):
                _num_arrays(v, f"{prefix}[{i}]", out)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def shannon_bits_per_sample(values: Sequence[float]) -> Optional[float]:
    """Empirical Shannon entropy of the symbol distribution, bits/sample.

    This is the floor for any *memoryless* lossless coder on this data. It is a
    floor, not a ceiling: context modelling can beat it on structured data.
    """
    n = len(values)
    if n == 0:
        return None
    counts: Dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def lzma_floor_ratio(values: Sequence[float], dtype: str = "d") -> Optional[float]:
    """Best ratio a strong general-purpose lossless coder reaches on this data.

    ``ratio`` here is raw_bytes / compressed_bytes, the same convention the
    "compression ratio" claims in these artifacts use. ``dtype`` is a struct
    format char: "d" = float64, "f" = float32, "q" = int64.
    """
    n = min(len(values), MAX_ARRAY_AUDIT)
    if n < 8:
        return None
    try:
        raw = struct.pack(f"<{n}{dtype}", *values[:n])
    except (struct.error, OverflowError):
        return None
    best = min(
        len(zlib.compress(raw, 9)),
        len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)),
    )
    if best == 0:
        return None
    return len(raw) / best


def entropy_floor_ratio(values: Sequence[float], bytes_per_sample: int) -> Optional[float]:
    """raw_bytes / (n * H) -- the memoryless-entropy ceiling on lossless ratio."""
    h = shannon_bits_per_sample(values)
    if h is None or h <= 0:
        return None
    return (bytes_per_sample * 8.0) / h


# ---------------------------------------------------------------------------
# hard bounds
# ---------------------------------------------------------------------------

# Highest PSNR/SNR that is numerically meaningful for IEEE-754 doubles.
# A double has ~15.95 decimal digits of precision => ~105.9 dB of dynamic range.
DOUBLE_MAX_DB = 105.9
FLOAT_MAX_DB = 51.2  # float32: ~7.22 digits => ~48.2 dB, plus margin
INT16_MAX_DB = 96.1  # 16-bit integer: 6.02*16 + 1.76
ABS_MAX_DB = 300.0  # anything above this is not a measurement


def audit_number(path: str, value: float, dtype_hint: str = "") -> Optional[Tuple[str, str, str]]:
    """Return (verdict, reason, detail) if this scalar is checkable, else None."""
    key = path.rsplit(".", 1)[-1].split("[")[0]

    if DB_KEY.search(key) and value > ABS_MAX_DB:
        return (
            Verdict.IMPOSSIBLE,
            "exceeds any physically representable dB figure",
            f"{value:g} dB -- no finite-precision measurement can exceed "
            f"{ABS_MAX_DB:g} dB (that would be a signal-to-error ratio of 1e15)",
        )

    if DB_KEY.search(key) and value > DOUBLE_MAX_DB:
        return (
            Verdict.SUSPECT,
            "above float64 dynamic range",
            f"{value:g} dB -- float64 tops out near {DOUBLE_MAX_DB:g} dB; "
            "higher means the reference and reconstruction are bit-identical "
            "(then report 'exact', not a dB number)",
        )

    if RATIO_KEY.search(key) and value > 1e6:
        return (
            Verdict.IMPOSSIBLE,
            "lossless ratio beyond what entropy permits",
            f"{value:g}:1 -- even all-zero data of this size has a bounded "
            "description length; a million-to-one 'lossless' ratio is a bug "
            "(usually an empty or truncated payload being compared)",
        )

    if LOSSLESS_KEY.search(key) and _is_num(value) and value not in (0.0, 1.0):
        return (
            Verdict.SUSPECT,
            "losslessness reported as a non-boolean quantity",
            f"{path} = {value:g} -- lossless is either exact (1) or not (0); "
            "a fractional value means the check is a threshold, not a proof",
        )

    return None


def audit_array(path: str, values: Sequence[float]) -> Dict[str, Any]:
    """Attach the real baselines to a data array so claims can be compared."""
    n = len(values)
    h = shannon_bits_per_sample(values)
    report: Dict[str, Any] = {
        "path": path,
        "n": n,
        "min": min(values),
        "max": max(values),
        "distinct_symbols": len({round(v, 12) for v in values}),
        "shannon_bits_per_sample": None if h is None else round(h, 6),
        "lzma9_ratio": None,
        "memoryless_entropy_ratio": None,
    }
    r = lzma_floor_ratio(values)
    if r is not None:
        report["lzma9_ratio"] = round(r, 4)
    e = entropy_floor_ratio(values, 8)
    if e is not None:
        report["memoryless_entropy_ratio"] = round(e, 4)
    return report


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


def _iter_inputs(paths: Iterable[str]) -> Iterable[str]:
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith(JSON_SUFFIXES) or f.endswith(LOG_SUFFIXES):
                        yield os.path.join(root, f)
        elif os.path.isfile(p):
            yield p
        else:
            yield p  # let the caller report it as missing


def audit_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)

    nums: List[Tuple[str, float]] = []
    _flat_nums(data, "", nums)

    flagged: List[Dict[str, str]] = []
    checked = 0
    for key, val in nums:
        checked += 1
        res = audit_number(key, val)
        if res:
            verdict, reason, detail = res
            flagged.append(
                {"path": key, "value": val, "verdict": verdict, "reason": reason, "detail": detail}
            )

    arrays: List[Tuple[str, Sequence[float]]] = []
    _num_arrays(data, "", arrays)
    baselines = [audit_array(k, v) for k, v in arrays]

    claims = [
        {"path": k, "value": v}
        for k, v in nums
        if CLAIM_KEY.search(k.rsplit(".", 1)[-1].split("[")[0])
    ]

    return {
        "file": path,
        "numeric_leaves": len(nums),
        "claim_like_values": len(claims),
        "claims": claims,
        "numbers_checked": checked,
        "flags": flagged,
        "array_baselines": baselines,
    }


def audit_log(path: str) -> Dict[str, Any]:
    hits: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            s = line.rstrip("\n")
            if re.search(r"\d", s) and CLAIM_KEY.search(s):
                hits.append(f"{path}:{i}: {s.strip()}")
    return {"file": path, "claim_lines": len(hits), "lines": hits}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="json/log files or directories")
    ap.add_argument("--json", action="store_true", help="emit machine-readable report")
    ap.add_argument("--max-lines", type=int, default=40, help="cap log lines shown")
    args = ap.parse_args(argv)

    reports: List[Dict[str, Any]] = []
    missing: List[str] = []

    for path in _iter_inputs(args.paths):
        if not os.path.exists(path):
            missing.append(path)
            continue
        try:
            if path.endswith(JSON_SUFFIXES):
                reports.append(audit_json(path))
            else:
                reports.append(audit_log(path))
        except (json.JSONDecodeError, OSError) as exc:
            missing.append(f"{path} ({exc})")

    impossible = [f for r in reports for f in r.get("flags", []) if f["verdict"] == Verdict.IMPOSSIBLE]
    suspect = [f for r in reports for f in r.get("flags", []) if f["verdict"] == Verdict.SUSPECT]

    if args.json:
        print(json.dumps({"reports": reports, "missing": missing}, indent=2))
        return 2 if not reports else (1 if impossible else 0)

    if not reports:
        print("NOTHING TO AUDIT.", file=sys.stderr)
        for m in missing:
            print(f"  missing: {m}", file=sys.stderr)
        print("Re-attach the artifacts or point me at the real path.", file=sys.stderr)
        return 2

    for r in reports:
        print(f"\n=== {r['file']} ===")
        if "numeric_leaves" in r:
            print(f"  numeric leaves: {r['numeric_leaves']}   claim-like: {r['claim_like_values']}")
            for c in r["claims"][:20]:
                print(f"    {c['path']} = {c['value']:g}")
            for b in r["array_baselines"][:8]:
                print(
                    f"    [array] {b['path']} n={b['n']} "
                    f"H={b['shannon_bits_per_sample']} bits/sample "
                    f"lzma9={b['lzma9_ratio']}x "
                    f"memoryless-ceiling={b['memoryless_entropy_ratio']}x"
                )
            for f in r["flags"]:
                print(f"    !! {f['verdict']}: {f['path']} = {f['value']:g} -- {f['reason']}")
                print(f"       {f['detail']}")
        else:
            print(f"  claim-bearing lines: {r['claim_lines']}")
            for ln in r["lines"][: args.max_lines]:
                print(f"    {ln}")

    print("\n--- summary ---")
    print(f"  files audited : {len(reports)}")
    print(f"  IMPOSSIBLE    : {len(impossible)}")
    print(f"  SUSPECT       : {len(suspect)}")
    if missing:
        print(f"  missing/unreadable: {len(missing)}")
    return 1 if impossible else 0


if __name__ == "__main__":
    raise SystemExit(main())
