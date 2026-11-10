# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Run the skill gate query matrix against a real cognee install, or the vendored reference.

Usage::

    python tools/audit_skill_gate.py [path/to/skill_gate_cases.json]      # real cognee if importable
    python tools/audit_skill_gate.py --reference                          # vendored reference only

Exit status is 0 when every row matches its expectation or is covered by a known upstream failure,
and 1 when behavior moved. The gate is pure regex with no I/O, so this costs milliseconds and can run
in any environment.

By default the real ``should_search_skills`` is used when cognee is importable, which is what makes
this an upstream regression check. ``--reference`` forces the vendored copy in ``skill_gate_ref.py``,
which keeps the matrix runnable with no dependencies at all; that mode validates the expectations
file rather than shipped behavior, so it says so loudly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

DEFAULT_CASES = Path(__file__).with_name("skill_gate_cases.json")


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def _real_gate() -> Callable[[str], Any]:
    from cognee.api.v1.recall.skill_gate import should_search_skills

    return should_search_skills


def _reference_gate() -> Callable[[str], Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from skill_gate_ref import should_search_skills

    return should_search_skills


def resolve_gate(reference: bool) -> tuple[Callable[[str], Any], str]:
    """Pick the gate under test and name it for the report."""
    if reference:
        return _reference_gate(), "vendored reference (skill_gate_ref.py)"
    try:
        return _real_gate(), "installed cognee"
    except ImportError:
        return _reference_gate(), "vendored reference (cognee is not importable here)"


def run_matrix(cases: list[dict[str, Any]], gate: Callable[[str], Any]) -> list[dict[str, Any]]:
    """Evaluate every case against ``should_search_skills`` and label the outcome.

    Labels:
        ``expected``   behavior matches the pinned expectation.
        ``known``      behavior differs but reproduces a documented upstream issue.
        ``unexpected`` behavior differs with no known issue attached. This is what fails CI.
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = gate(case["query"])
        fired = bool(result.fired)
        matches = fired == case["expected"]
        if matches:
            status = "expected"
        elif case.get("known_failure"):
            status = "known"
        else:
            status = "unexpected"
        rows.append(
            {
                "query": case["query"],
                "expected": case["expected"],
                "fired": fired,
                "score": float(result.score),
                "matched": list(result.matched),
                "status": status,
                "known_failure": case.get("known_failure"),
                "note": case.get("note", ""),
            }
        )
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="?", default=str(DEFAULT_CASES), help="case matrix JSON")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="force the vendored reference gate instead of an installed cognee",
    )
    args = parser.parse_args(argv[1:])

    cases = load_cases(Path(args.cases))
    gate, source = resolve_gate(args.reference)
    rows = run_matrix(cases, gate)
    print(f"gate under test: {source}\n")

    for row in rows:
        print(
            f"[{row['status']:>10}] fired={str(row['fired']):<5} "
            f"expected={str(row['expected']):<5} score={row['score']:.1f} :: {row['query']}"
        )

    unexpected = [row for row in rows if row["status"] == "unexpected"]
    known = [row for row in rows if row["status"] == "known"]
    print(
        f"\n{len(rows)} cases | {len(rows) - len(unexpected) - len(known)} as expected | "
        f"{len(known)} known upstream | {len(unexpected)} unexpected"
    )
    if known:
        issues = sorted({row["known_failure"] for row in known})
        print(f"known upstream issues reproduced: {', '.join('#' + issue for issue in issues)}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
