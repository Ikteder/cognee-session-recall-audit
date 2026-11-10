# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Skill gate matrix as a test.

Runs against a real cognee install when one is importable, and against the pinned local copy of the
rule set otherwise. Rows carrying a ``known_failure`` label are allowed to deviate from the
expectation; any other deviation fails. That way the suite documents current upstream behavior
without pretending it is correct, and it starts failing the moment that behavior moves for an
unexplained reason.

The two gates are kept separate on purpose. The reference copy only validates the expectations file;
only the installed cognee validates shipped behavior. A run that silently falls back would otherwise
report a green suite while testing nothing upstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_skill_gate import DEFAULT_CASES, load_cases, resolve_gate, run_matrix  # noqa: E402

CASES = load_cases(DEFAULT_CASES)

try:
    REAL_GATE, GATE_SOURCE = resolve_gate(reference=False)
    USING_REAL_COGNEE = "installed cognee" in GATE_SOURCE
except Exception:  # pragma: no cover - only hit when even the reference copy fails to import
    REAL_GATE, GATE_SOURCE, USING_REAL_COGNEE = None, "unavailable", False

REFERENCE_GATE = resolve_gate(reference=True)[0]


def _gate():
    """The gate under test plus a label, matching whichever install is present."""
    return (REAL_GATE, GATE_SOURCE) if USING_REAL_COGNEE else (REFERENCE_GATE, "pinned reference copy")


def test_matrix_has_no_unexpected_rows():
    gate, source = _gate()
    rows = run_matrix(CASES, gate)
    unexpected = [row for row in rows if row["status"] == "unexpected"]

    assert not unexpected, f"{source}: gate behavior moved: {unexpected}"


@pytest.mark.parametrize("case", CASES, ids=[c["query"] for c in CASES])
def test_case_is_expected_or_a_known_upstream_failure(case):
    gate, source = _gate()
    row = run_matrix([case], gate)[0]

    if row["status"] == "unexpected":
        pytest.fail(f"{source}: {case['query']!r} fired={row['fired']} expected={case['expected']}")
    if case["expected"] and not row["known_failure"]:
        assert row["score"] >= 1.0, "an expected procedural match must clear the threshold"


def test_known_failures_reproduce_the_documented_upstream_bug():
    """The negation-leak rows must still leak while cognee #5647 is open.

    The bug is that a clause-scoped negation cancels procedural signals in later clauses. English
    clause breaks are handled, so the English rows in the matrix fire and only their notes document
    what a correct gate would do; the Chinese rows still leak. Pinning the count keeps the reference
    copy from quietly growing clause support that upstream does not have.
    """
    gate, source = _gate()
    leaking = [
        case["query"]
        for case in CASES
        if case.get("known_failure") and not gate(case["query"]).fired
    ]

    assert len(leaking) == 1, f"{source}: expected exactly one still-leaking row, got {leaking}"
    assert all(any(ord(ch) > 127 for ch in q) for q in leaking), (
        f"{source}: only CJK rows should still leak; got {leaking}"
    )


def test_reference_copy_stays_in_sync_with_the_matrix():
    """The vendored rule set is a copy of upstream logic, so it has to satisfy the same matrix.

    This is what catches drift between ``skill_gate_ref.py`` and the expectations file when no real
    cognee install is present.
    """
    rows = run_matrix(CASES, REFERENCE_GATE)
    unexpected = [row for row in rows if row["status"] == "unexpected"]

    assert not unexpected, f"reference copy disagrees with the matrix: {unexpected}"


@pytest.mark.skipif(not USING_REAL_COGNEE, reason="cognee is not installed in this environment")
def test_real_cognee_gate_is_the_one_being_measured():
    """Guards against a silent fallback making an upstream run look green."""
    assert USING_REAL_COGNEE


def test_same_clause_negation_is_suppressed():
    gate, _ = _gate()
    for query in ("don't walk me through it", "I can't set up the cluster", "不要带我一步步操作"):
        assert not gate(query).fired, f"{query!r} must not fire"


def test_procedural_queries_fire():
    gate, _ = _gate()
    for query in ("how do I fix it?", "walk me through the deploy", "step-by-step fix?", "怎么排查？"):
        assert gate(query).fired, f"{query!r} must fire"


def test_declarative_queries_do_not_fire():
    gate, _ = _gate()
    for query in ("what is Kubernetes?", "the deployment rolled back"):
        assert not gate(query).fired, f"{query!r} must not fire"
