# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Public API of the audit kit."""

from audit.corpus import (
    CURATED_LESSONS,
    DATASET_NAME,
    DOCUMENT_TEXTS,
    PROBE_QUERIES,
    SESSION_ID,
    SESSION_TURNS,
)
from audit.distill_audit import (
    DistillAudit,
    LessonSpan,
    TurnCoverage,
    audit_curated_lessons,
    lesson_spans,
    lesson_texts,
)
from audit.recall_probe import (
    RecallComparison,
    compare_recall_context,
    extract_context,
    measure_session_lift,
)
from audit.report import AuditReport, format_report, summarize_lift

__all__ = [
    "AuditReport",
    "CURATED_LESSONS",
    "DATASET_NAME",
    "DOCUMENT_TEXTS",
    "DistillAudit",
    "LessonSpan",
    "PROBE_QUERIES",
    "RecallComparison",
    "SESSION_ID",
    "SESSION_TURNS",
    "TurnCoverage",
    "audit_curated_lessons",
    "compare_recall_context",
    "extract_context",
    "format_report",
    "lesson_spans",
    "lesson_texts",
    "measure_session_lift",
    "summarize_lift",
]
