"""Self-check for the audit kit.

Runs without cognee installed. The two probes are exercised against stand-in recall and curation
callables that reproduce the shapes of the real ones, so the measurement code itself is verified
even in a bare environment, and the skill gate matrix runs against the local copy of the rule set
when no real cognee install is importable.

Run with::

    python -m selfcheck          # or: python selfcheck.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from audit.corpus import (
    CURATED_LESSONS,
    DATASET_NAME,
    PROBE_QUERIES,
    SESSION_ID,
    SESSION_TURNS,
)
from audit.distill_audit import audit_curated_lessons, lesson_texts, turn_lookup
from audit.recall_probe import compare_recall_context, extract_context, measure_session_lift
from audit.report import AuditReport, format_report, summarize_lift

HERE = Path(__file__).resolve().parent


class FakeRecall:
    """Stand-in for ``cognee.recall`` with ``only_context=True``.

    Reproduces the documented return shape (list of dicts tagged with ``text`` and ``source``) and
    models the session layer as an extra entry carrying distilled lessons. That is exactly the
    behavior under audit, so a probe that reads this correctly will read the real one correctly.
    """

    def __init__(self, graph_rows: dict[str, list[str]], session_rows: dict[str, list[str]]):
        self.graph_rows = graph_rows
        self.session_rows = session_rows
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append(kwargs)
        query = kwargs["query_text"]
        rows = [
            {"text": text, "source": "graph", "search_type": "GRAPH_COMPLETION"}
            for text in self.graph_rows.get(query, [])
        ]
        if kwargs.get("session_id"):
            rows += [
                {"text": text, "source": "session_context", "search_type": "SESSION_CONTEXT"}
                for text in self.session_rows.get(query, [])
            ]
        return rows


def _graph_and_session_rows() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    runbook = (
        "Pod stuck in CrashLoopBackOff: check the container exit code with kubectl describe pod, "
        "then read the previous logs. A repeated 137 means an OOMKill; raise the memory limit and "
        "restart the deployment."
    )
    shared = {query: [runbook] for query in PROBE_QUERIES}
    session = {
        PROBE_QUERIES[0]: [CURATED_LESSONS[0]],
        PROBE_QUERIES[2]: [CURATED_LESSONS[1]],
    }
    return shared, session


def check_session_lift() -> tuple[list[Any], dict[str, int]]:
    graph_rows, session_rows = _graph_and_session_rows()
    recall = FakeRecall(graph_rows, session_rows)

    comparisons = asyncio.run(
        measure_session_lift(
            recall,
            PROBE_QUERIES,
            datasets=(DATASET_NAME,),
            session_id=SESSION_ID,
        )
    )

    assert len(comparisons) == len(PROBE_QUERIES), "one comparison per probed query"
    assert all(call["only_context"] is True for call in recall.calls), (
        "the probe must never trigger LLM generation"
    )
    assert len(recall.calls) == 2 * len(PROBE_QUERIES), "every query runs a graph leg and a session leg"
    assert [call.get("session_id") for call in recall.calls[1::2]] == [SESSION_ID] * len(PROBE_QUERIES), (
        "the second leg of every pair must carry the session id so the session layer is exercised"
    )

    lifted = [item for item in comparisons if item.lifted]
    assert len(lifted) == 2, f"expected 2 lifted queries, got {len(lifted)}"
    assert all(item.delta_chars > 0 for item in lifted), "session leg must add content"
    assert extract_context(None) == "", "None result must degrade to empty context"

    unchanged = [item for item in comparisons if not item.lifted and not item.error]
    assert len(unchanged) == 1, "one query is answerable from the graph alone"

    # A session leg that errors out is unmeasured, not a contribution. It must be reported as an error
    # and never counted as lift.
    def broken_recall(**kwargs: Any) -> list[dict[str, str]]:
        if kwargs.get("session_id"):
            raise RuntimeError("session store unavailable")
        return [{"text": _graph_and_session_rows()[0][PROBE_QUERIES[0]][0], "source": "graph"}]

    failed = asyncio.run(compare_recall_context(broken_recall, PROBE_QUERIES[0]))
    assert failed.error and "session leg failed" in failed.notes[0], "failure must be recorded"
    assert not failed.lifted, "an unmeasured session leg must never count as lift"

    return comparisons, summarize_lift(comparisons)


def check_distillation() -> Any:
    audit = audit_curated_lessons(CURATED_LESSONS, SESSION_TURNS)

    assert audit.lessons == list(CURATED_LESSONS), "lessons audited as given"
    assert not audit.unsupported_spans, (
        f"unsupported spans: {[s.span for s in audit.unsupported_spans]}"
    )
    assert audit.coverage == 1.0, f"expected full coverage, got {audit.coverage:.3f}"

    assistant_turns = [t for t in audit.turns if t.role == "assistant"]
    assert len(assistant_turns) == 2, "both assistant turns must be tracked"
    assert all(t.covered for t in assistant_turns), "every assistant turn must contribute"
    assert all(t.coverage == 1.0 for t in assistant_turns), (
        "expected every assistant turn fully covered, got "
        f"{[(t.role, t.index, t.coverage) for t in assistant_turns]}"
    )

    # Upstream does not distill question text into lessons ("User asked about X" is explicitly a
    # rejected breadcrumb), so user turns stay unscored by design.
    user_turns = [t for t in audit.turns if t.role == "user"]
    assert all(not t.covered and t.coverage == 0.0 for t in user_turns), (
        "user turns must not be scored as lesson sources"
    )

    # A lesson that quotes something no cited turn contains has to be caught.
    broken = list(CURATED_LESSONS) + [
        "Always restart the deployment before editing ConfigMaps. [U1][A1]"
    ]
    broken_audit = audit_curated_lessons(broken, SESSION_TURNS)
    assert len(broken_audit.unsupported_spans) == 1, (
        f"expected 1 unsupported span, got {[s.span for s in broken_audit.unsupported_spans]}"
    )

    # A lesson with no citation cannot be traced and has to be reported.
    uncited = audit_curated_lessons(["Restart the deployment after editing ConfigMaps."], SESSION_TURNS)
    assert uncited.notes and "no [U#]/[A#] citation" in uncited.notes[0]

    # A turn that contributes nothing shows up as lost, not silently covered.
    extra_turns = list(SESSION_TURNS) + [("assistant", "Never edit ConfigMaps by hand. Ever.")]
    partial = audit_curated_lessons(CURATED_LESSONS, extra_turns)
    assert partial.uncovered_turns, "an unused turn must be reported as uncovered"

    lookup = turn_lookup(SESSION_TURNS)
    assert ("U", 1) in lookup and ("A", 1) in lookup, "turn keys must follow curation numbering"
    assert lesson_texts([CURATED_LESSONS[0]]) == [CURATED_LESSONS[0]], "string lessons pass through"

    return audit


def check_skill_gate() -> list[dict[str, Any]]:
    from tools.audit_skill_gate import load_cases, resolve_gate, run_matrix

    # selfcheck runs with no dependencies installed, so it pins the reference gate. The same matrix
    # against an installed cognee is what validates upstream behavior; tools/audit_skill_gate.py does
    # that automatically whenever cognee is importable.
    gate, source = resolve_gate(reference=True)
    rows = run_matrix(load_cases(), gate)
    assert len(rows) >= 12, "the matrix must cover the shipped cases"

    unexpected = [row for row in rows if row["status"] == "unexpected"]
    assert not unexpected, f"unexpected gate behavior: {unexpected}"

    fired = [row for row in rows if row["expected"] and row["fired"]]
    suppressed = [row for row in rows if not row["expected"] and not row["fired"]]
    assert len(fired) >= 4 and len(suppressed) >= 4, "matrix must exercise both directions"

    known = [row for row in rows if row["status"] == "known"]
    assert known, "the matrix should reproduce the known upstream negation leak"
    print(f"skill gate: {source}")
    return rows


def build_report() -> AuditReport:
    comparisons, summary = check_session_lift()
    distillation = check_distillation()
    gate_rows = check_skill_gate()

    try:
        from cognee import __version__ as version
    except Exception:
        version = None

    report = AuditReport(
        lift=comparisons,
        distillation=distillation,
        skill_gate=gate_rows,
        cognee_version=version,
    )
    assert summary["lifted"] == 2 and summary["unchanged"] == 1 and summary["errored"] == 0
    assert not report.probe_errors, "self-check must measure every session leg"
    assert report.ok, "self-check report must pass"
    return report


def main() -> int:
    report = build_report()
    # Windows consoles default to a legacy code page. Reconfigure stdout so the table never dies on an
    # encoding error after the measurements have already been paid for.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    print(format_report(report))
    print("\nselfcheck: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
