# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Offline measurement harness for the two probes, without cognee installed.

The probes are written against documented API shapes and this harness feeds them those shapes, so
the measurement code is verified on any machine. It also documents the contract: what a recall
result looks like in ``only_context=True`` mode, and what curation hands back.
"""

from __future__ import annotations

import asyncio

import pytest

from audit.corpus import CURATED_LESSONS, DATASET_NAME, PROBE_QUERIES, SESSION_ID, SESSION_TURNS
from audit.distill_audit import audit_curated_lessons
from audit.recall_probe import compare_recall_context, extract_context, measure_session_lift
from audit.report import AuditReport, format_report, summarize_lift
from selfcheck import FakeRecall, _graph_and_session_rows


def test_only_context_is_always_requested():
    """The probe must never ask recall to generate an answer."""
    graph_rows, session_rows = _graph_and_session_rows()
    recall = FakeRecall(graph_rows, session_rows)

    asyncio.run(measure_session_lift(recall, PROBE_QUERIES[:1], session_id=SESSION_ID))

    assert len(recall.calls) == 2
    for call in recall.calls:
        assert call["only_context"] is True
        assert call["datasets"] == [DATASET_NAME]
        assert call["top_k"] == 15


def test_session_leg_carries_the_session_id():
    graph_rows, session_rows = _graph_and_session_rows()
    recall = FakeRecall(graph_rows, session_rows)

    comparison = asyncio.run(compare_recall_context(recall, PROBE_QUERIES[0]))

    assert "session_id" not in recall.calls[0], "first leg is the graph-only baseline"
    assert recall.calls[1]["session_id"] == SESSION_ID
    assert comparison.lifted and comparison.delta_chars > 0


def test_queries_without_session_content_are_reported_as_unchanged():
    graph_rows, session_rows = _graph_and_session_rows()
    recall = FakeRecall(graph_rows, session_rows)

    comparisons = asyncio.run(measure_session_lift(recall, PROBE_QUERIES))
    summary = summarize_lift(comparisons)

    assert summary == {"total": 3, "lifted": 2, "unchanged": 1, "errored": 0}
    quiet = [item for item in comparisons if not item.lifted]
    assert quiet[0].query == PROBE_QUERIES[1]
    assert quiet[0].identical and quiet[0].delta_chars == 0
    assert quiet[0].error is None, "the unchanged row was measured, it did not fail"


def test_session_leg_without_a_session_is_skipped_not_silently_equal():
    graph_rows, session_rows = _graph_and_session_rows()
    recall = FakeRecall(graph_rows, session_rows)

    comparison = asyncio.run(compare_recall_context(recall, PROBE_QUERIES[0], session_id=None))

    assert len(recall.calls) == 1
    assert comparison.notes and "session leg skipped" in comparison.notes[0]


def test_a_broken_session_leg_is_reported_not_raised():
    def broken_recall(**kwargs):
        if kwargs.get("session_id"):
            raise RuntimeError("session store unavailable")
        return [{"text": "graph only", "source": "graph"}]

    comparison = asyncio.run(compare_recall_context(broken_recall, PROBE_QUERIES[0]))

    assert not comparison.lifted, "an unmeasured session leg must never count as lift"
    assert comparison.error == "RuntimeError: session store unavailable"
    assert any("session leg failed" in note for note in comparison.notes)
    # Errored rows are their own bucket so a red CI run cannot be mistaken for "no memory benefit".
    assert summarize_lift([comparison]) == {"total": 1, "lifted": 0, "unchanged": 0, "errored": 1}


def test_extract_context_reads_both_result_shapes():
    from_dict = extract_context([{"text": "from a dict"}])
    from_object = extract_context([type("Entry", (), {"text": "from an object"})()])
    from_plain = extract_context("plain string")

    assert from_dict == "from a dict"
    assert from_object == "from an object"
    assert from_plain == "plain string"
    assert extract_context(None) == ""


def test_distillation_reports_full_coverage_for_a_faithful_curator():
    audit = audit_curated_lessons(CURATED_LESSONS, SESSION_TURNS)

    assert audit.coverage == 1.0
    assert audit.unsupported_spans == []
    assert audit.uncovered_turns == []
    assistant = [t for t in audit.turns if t.role == "assistant"]
    assert len(audit.spans) == len(CURATED_LESSONS)
    assert all(t.coverage == 1.0 for t in assistant)


def test_user_turns_are_not_scored_as_lesson_sources():
    """Upstream rejects breadcrumbs like "User asked about X", so question text stays unscored."""
    audit = audit_curated_lessons(CURATED_LESSONS, SESSION_TURNS)
    users = [t for t in audit.turns if t.role == "user"]

    assert users, "the fixture has user turns"
    assert all(not t.covered and t.coverage == 0.0 for t in users)
    # Adding a turn no lesson quotes has to show up as uncovered, not silently pass.
    extra = audit_curated_lessons(CURATED_LESSONS, list(SESSION_TURNS) + [("assistant", "Never edit ConfigMaps by hand.")])
    assert len(extra.uncovered_turns) == 1


def test_distillation_flags_a_span_no_cited_turn_contains():
    broken = ["Never restart the pods before checking logs. [U1][A1]"]
    audit = audit_curated_lessons(broken, SESSION_TURNS)

    assert len(audit.unsupported_spans) == 1
    assert audit.unsupported_spans[0].turn_index == 1
    assert audit.coverage < 1.0


def test_distillation_flags_uncited_and_unused():
    uncited = audit_curated_lessons(["A lesson with no citation at all."], SESSION_TURNS)
    assert uncited.spans == [] and uncited.notes

    empty = audit_curated_lessons([], SESSION_TURNS)
    assert empty.notes == ["no lessons to audit"] and empty.coverage == 0.0


def test_report_renders_and_marks_the_run_passed():
    graph_rows, session_rows = _graph_and_session_rows()
    recall = FakeRecall(graph_rows, session_rows)

    report = AuditReport(
        lift=asyncio.run(measure_session_lift(recall, PROBE_QUERIES)),
        distillation=audit_curated_lessons(CURATED_LESSONS, SESSION_TURNS),
        skill_gate=[{"query": "how do I fix it?", "expected": True, "fired": True, "status": "expected"}],
    )

    text = format_report(report)
    assert report.ok
    assert "verdict: PASS" in text
    assert "assistant coverage: 100%" in text
    assert report.as_dict()["distillation"]["coverage"] == 1.0


def test_report_fails_when_a_lesson_is_unsupported():
    report = AuditReport(
        distillation=audit_curated_lessons(["Ship it to production on Fridays. [U1][A1]"], SESSION_TURNS)
    )

    assert not report.ok
    assert "unsupported span" in format_report(report)


def test_report_fails_when_a_probe_errored():
    """An unmeasured session leg is a broken harness, not evidence that memory adds nothing."""
    failed = asyncio.run(
        compare_recall_context(
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("db down")) if kwargs.get("session_id") else [],
            PROBE_QUERIES[0],
        )
    )
    report = AuditReport(
        lift=[failed],
        distillation=audit_curated_lessons(CURATED_LESSONS, SESSION_TURNS),
    )

    assert report.probe_errors == [PROBE_QUERIES[0]]
    assert not report.ok
    assert "error" in format_report(report)
