# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Structured audit results and a plain-text summary.

Kept separate from the probes so the measurement code has no printing concerns and the report can
be consumed as data by CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from audit.distill_audit import DistillAudit
from audit.recall_probe import RecallComparison


@dataclass
class AuditReport:
    """Everything the two probes measured, in one object.

    Attributes:
        lift: One comparison per probed query.
        distillation: Coverage audit of the curated lessons, or ``None`` when curation produced
            nothing to audit.
        skill_gate: Rows from the skill gate matrix, each a plain dict.
        cognee_version: Version string of the cognee install that was measured, when available.
    """

    lift: list[RecallComparison] = field(default_factory=list)
    distillation: DistillAudit | None = None
    skill_gate: list[dict[str, Any]] = field(default_factory=list)
    cognee_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cognee_version": self.cognee_version,
            "session_lift": [item.as_dict() for item in self.lift],
            "distillation": self.distillation.as_dict() if self.distillation else None,
            "skill_gate": list(self.skill_gate),
        }

    @property
    def queries_lifted(self) -> int:
        return sum(1 for item in self.lift if item.lifted)

    @property
    def probe_errors(self) -> list[str]:
        """Queries whose session leg could not be measured at all."""
        return [item.query for item in self.lift if item.error]

    @property
    def ok(self) -> bool:
        """True when every probe found something and no unsupported spans were reported.

        A zero-lift result is not automatically a failure: some queries genuinely are answerable
        from the graph alone. What must never happen silently is an unsupported lesson span, a
        distillation run that produced no lessons at all, or a session leg that errored out and left
        the comparison unmeasured.
        """
        if self.distillation is None or not self.distillation.lessons:
            return False
        if self.distillation.unsupported_spans:
            return False
        if self.probe_errors:
            return False
        unexpected = [row for row in self.skill_gate if row.get("status") == "unexpected"]
        return not unexpected


def format_report(report: AuditReport) -> str:
    """Render the report as a fixed-width text block suitable for stdout."""
    lines: list[str] = []
    header = "cognee session memory audit"
    if report.cognee_version:
        header += f"  (cognee {report.cognee_version})"
    lines.append(header)
    lines.append("=" * len(header))

    lines.append("")
    lines.append("1. Session lift (retrieval context, no LLM)")
    lines.append("-" * 48)
    if not report.lift:
        lines.append("  no queries probed")
    for item in report.lift:
        if item.error:
            verdict = "error"
        elif item.lifted:
            verdict = "lifted"
        else:
            verdict = "no change"
        # The probe records the real query string. Truncate for the table rather than rewording it, so
        # what is printed stays traceable to the recall() call that produced the row.
        query = item.query if len(item.query) <= 60 else item.query[:57] + "..."
        lines.append(
            f"  [{verdict:>8}] {query}\n"
            f"             graph {len(item.graph_context):>5} chars | "
            f"session {len(item.session_context):>5} chars | delta {item.delta_chars:+d}"
        )
        for note in item.notes:
            lines.append(f"             note: {note}")

    lines.append("")
    lines.append("2. Distillation coverage (curated lessons vs source turns)")
    lines.append("-" * 48)
    audit = report.distillation
    if audit is None:
        lines.append("  not run")
    else:
        lines.append(f"  lessons: {len(audit.lessons)} | assistant coverage: {audit.coverage:.0%}")
        for turn in audit.turns:
            if turn.role != "assistant":
                # Curation does not distill question text into lessons, so user turns are listed for
                # completeness but never scored.
                lines.append(f"  [ n/a ] {turn.role} #{turn.index}: not a lesson source")
                continue
            flag = "kept" if turn.covered else "LOST"
            lines.append(f"  [{flag:>4}] {turn.role} #{turn.index}: {turn.coverage:.0%}")
            for lost in turn.as_dict()["lost_sentences"]:
                lines.append(f"         unrepresented: {lost}")
        for span in audit.unsupported_spans:
            lines.append(
                f"  unsupported span in lesson {span.lesson_index} "
                f"(cites [{span.role}{span.turn_index}]): {span.span}"
            )
        for note in audit.notes:
            lines.append(f"  note: {note}")

    lines.append("")
    lines.append("3. Skill gate matrix")
    lines.append("-" * 48)
    if not report.skill_gate:
        lines.append("  not run")
    for row in report.skill_gate:
        status = row.get("status", "?")
        query = str(row.get("query", ""))
        if len(query) > 60:
            query = query[:57] + "..."
        lines.append(
            f"  [{status:>10}] fired={row.get('fired')} expected={row.get('expected')} "
            f"score={row.get('score')} :: {query}"
        )

    lines.append("")
    lines.append(f"verdict: {'PASS' if report.ok else 'FAIL'}")
    return "\n".join(lines)


def summarize_lift(comparisons: Sequence[RecallComparison]) -> dict[str, int]:
    """Bucket queries into lifted, unchanged and errored; handy for assertions in CI."""
    return {
        "total": len(comparisons),
        "lifted": sum(1 for item in comparisons if item.lifted),
        "unchanged": sum(1 for item in comparisons if not item.lifted and not item.error),
        "errored": sum(1 for item in comparisons if item.error),
    }
