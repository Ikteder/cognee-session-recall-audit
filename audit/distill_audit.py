# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Distillation coverage audit.

``cognee.session.distill_session()`` curates session turns into lessons and then hands them to a
writer LLM that persists them. The writer is where content gets dropped or reworded, and the only
upstream example of the flow (`examples/guides/session_distillation.py`) checks one snack choice
after the fact.

This probe stops after curation and audits the curated lessons directly:

* every quoted span in a lesson must appear verbatim in the turns it cites;
* every input turn must contribute at least one supported span, or it is reported as lost.

That gives a coverage number plus an explicit loss list, with no LLM involved. Curation itself is
deterministic (rule based), so these numbers are stable across runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Matches the citation markers curation writes into lesson bodies: "[U1]" / "[A3]".
_CITATION = re.compile(r"\[(?P<role>[UA])(?P<index>\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A span shorter than this carries no signal; matching it would inflate coverage.
MIN_SPAN_CHARS = 12


@dataclass
class LessonSpan:
    """One verbatim fragment of a lesson and where it came from.

    Attributes:
        lesson_index: Position of the lesson in the curated list (0 based).
        role: ``"U"`` for a user turn, ``"A"`` for an assistant turn.
        turn_index: 1-based turn index cited by the lesson.
        span: The fragment being checked.
        supported: Whether ``span`` appears verbatim in the cited turn.
    """

    lesson_index: int
    role: str
    turn_index: int
    span: str
    supported: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "lesson": self.lesson_index,
            "role": self.role,
            "turn": self.turn_index,
            "span": self.span,
            "supported": self.supported,
        }


@dataclass
class TurnCoverage:
    """How much of one input turn survived into the curated lessons.

    Attributes:
        role: ``"user"`` or ``"assistant"``.
        index: 1-based turn index.
        text: The original turn text.
        matched_spans: Supported spans attributed to this turn.
    """

    role: str
    index: int
    text: str
    matched_spans: list[str] = field(default_factory=list)

    @property
    def covered(self) -> bool:
        return bool(self.matched_spans)

    @property
    def coverage(self) -> float:
        """Fraction of this turn's sentences that a supported span traces back to.

        Only assistant turns are scored. Curation extracts reusable facts from the answers, not from
        question text (upstream asks for generalizable lessons and rejects session breadcrumbs like
        "User asked about X"), so user turns stay at 0.0 by design and the aggregate averages
        assistant turns only.

        A sentence counts as covered when some supported span quotes it, contains it, sits inside it,
        or shares content words with it. Curation re-segments and compresses, so scoring exact
        substrings alone would report a faithful run as lossy and make the metric useless.
        """
        if self.role != "assistant":
            return 0.0
        sentences = split_sentences(self.text)
        if not sentences:
            return 0.0
        covered_count = sum(1 for sentence in sentences if _is_kept(sentence, self.matched_spans))
        return covered_count / len(sentences)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "turn": self.index,
            "covered": self.covered,
            "coverage": round(self.coverage, 3),
            "matched_spans": list(self.matched_spans),
            "lost_sentences": [
                sentence for sentence in split_sentences(self.text)
                if not _is_kept(sentence, self.matched_spans)
            ],
        }


@dataclass
class DistillAudit:
    """Aggregate result of auditing a curated lesson set against its input turns.

    Attributes:
        lessons: The lesson texts that were audited.
        spans: Every span checked, with its support verdict.
        turns: Per-turn coverage records, in input order.
        notes: Anything unusual (empty lesson list, no citations, and so on).
    """

    lessons: list[str] = field(default_factory=list)
    spans: list[LessonSpan] = field(default_factory=list)
    turns: list[TurnCoverage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def unsupported_spans(self) -> list[LessonSpan]:
        """Spans that a lesson asserts but no cited turn contains."""
        return [span for span in self.spans if not span.supported]

    @property
    def uncovered_turns(self) -> list[TurnCoverage]:
        """Assistant turns that contributed nothing to any lesson.

        User turns are excluded: upstream curation extracts reusable facts from answers and rejects
        breadcrumbs like "User asked about X", so an unquoted user turn is normal.
        """
        return [turn for turn in self.turns if turn.role == "assistant" and not turn.covered]

    @property
    def coverage(self) -> float:
        """Mean coverage over the assistant turns that contain text.

        User turns are excluded because curation does not distill question text into lessons, so
        averaging them in would pin the number near zero no matter how good the distillation was.
        """
        scored = [turn for turn in self.turns if turn.role == "assistant" and turn.text.strip()]
        if not scored:
            return 0.0
        return sum(turn.coverage for turn in scored) / len(scored)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lessons": list(self.lessons),
            "coverage": round(self.coverage, 3),
            "spans_total": len(self.spans),
            "spans_unsupported": [span.as_dict() for span in self.unsupported_spans],
            "turns": [turn.as_dict() for turn in self.turns],
            "notes": list(self.notes),
        }


def split_sentences(text: str) -> list[str]:
    """Split turn text into sentences, dropping whitespace-only fragments."""
    return [piece.strip() for piece in _SENTENCE_SPLIT.split(text.strip()) if piece.strip()]


def lesson_spans(lesson: str) -> list[tuple[str, int]]:
    """Extract ``(role, turn_index)`` citations from one lesson body.

    The final ``[U1][A3]`` block is what curation writes; a lesson with no citation cannot be
    traced, so it yields no spans and the caller reports that.
    """
    return [(match.group("role"), int(match.group("index"))) for match in _CITATION.finditer(lesson)]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _is_kept(sentence: str, spans: Iterable[str]) -> bool:
    """True when a turn sentence shares content words with some supported span.

    Either direction of containment counts as well: a longer span that quotes the sentence, or a
    compressed span that sits inside it. Content word overlap is what makes a compressed quote like
    "A repeated 137 means an OOMKill" cover its longer source sentence.
    """
    normalized = _normalize(sentence)
    for span in spans:
        span_text = _normalize(span)
        if normalized in span_text or span_text in normalized:
            return True
        if _content_tokens(sentence) & _content_tokens(span):
            return True
    return False


def _span_survives(span: str, source: str) -> bool:
    """True when every content word of a lesson span appears in the turn it cites.

    Curation re-segments and compresses: a curator that writes "A repeated 137 means an OOMKill."
    for the source sentence "A repeated 137 means an OOMKill, so raise the memory limit" kept the
    wording and dropped a clause, which is faithful. A span that introduces content words the source
    never used ("Never restart", "Always edit") is an invention, and that is what this audit exists
    to catch, so it stays unsupported.

    Content words are chosen over exact containment on purpose: containment breaks on ordinary
    curator re-segmentation and would report a faithful run as lossy, which makes the metric useless.
    """
    source_tokens = _content_tokens(source)
    span_tokens = _content_tokens(span)
    if not span_tokens:
        return False
    return span_tokens.issubset(source_tokens)


def _content_tokens(text: str) -> set[str]:
    """Lowercased word tokens with punctuation and stop words removed."""
    words = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
    return {word for word in words if word not in _STOP_WORDS}


# Function words that carry no traceable content. Kept small on purpose: the wider the stop list, the
# more of an invented span slips through as supported.
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "so", "then", "to", "of", "in", "on", "for", "with",
        "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "as", "at", "by",
        "from", "after", "before", "into", "than", "when", "while", "because", "if", "we", "you",
    }
)


def turn_lookup(turns: Sequence[tuple[str, str]]) -> dict[tuple[str, int], str]:
    """Map curation's ``(role, 1-based index)`` keys onto the original turn text.

    ``turns`` is a sequence of ``(role, content)`` pairs in conversation order, where role is
    ``"user"`` or ``"assistant"``. Curation numbers user and assistant turns separately.
    """
    counters = {"U": 0, "A": 0}
    short = {"user": "U", "assistant": "A"}
    lookup: dict[tuple[str, int], str] = {}
    for role, content in turns:
        key = short.get(role)
        if key is None:
            continue
        counters[key] += 1
        lookup[(key, counters[key])] = content
    return lookup


def audit_curated_lessons(
    lessons: Sequence[str],
    turns: Sequence[tuple[str, str]],
) -> DistillAudit:
    """Audit curated lesson texts against the session turns they were built from.

    Args:
        lessons: Lesson bodies in curation order. Accepts the strings produced by
            :func:`cognee.modules.session_distillation.models.distill_documents` or plain text.
        turns: The session turns as ``(role, content)`` pairs in conversation order.

    Returns:
        A :class:`DistillAudit` with per-span support verdicts and per-turn coverage.
    """
    audit = DistillAudit(lessons=[str(lesson) for lesson in lessons])
    lookup = turn_lookup(turns)

    roles = {"user": "user", "assistant": "assistant"}
    for role in ("user", "assistant"):
        prefix = roles[role]
        index = 0
        for content in (text for r, text in turns if r == prefix):
            index += 1
            audit.turns.append(TurnCoverage(role=role, index=index, text=content))

    if not audit.lessons:
        audit.notes.append("no lessons to audit")
        return audit

    by_key = {(turn.role, turn.index): turn for turn in audit.turns}

    for lesson_index, lesson in enumerate(audit.lessons):
        citations = lesson_spans(lesson)
        if not citations:
            audit.notes.append(f"lesson {lesson_index}: no [U#]/[A#] citation, cannot be traced")
            continue

        body = _CITATION.sub("", lesson).strip()
        cited = [(role, turn_index) for role, turn_index in dict.fromkeys(citations)]
        # A sentence is a span of the whole lesson, not of one citation. Attributing per citation
        # would double count it and flag every sentence against turns that never said it.
        sources = []
        for role, turn_index in cited:
            source = lookup.get((role, turn_index))
            if source is None:
                audit.notes.append(
                    f"lesson {lesson_index}: cites [{role}{turn_index}] which is not an input turn"
                )
                continue
            sources.append((role, turn_index, source))

        sentences = split_sentences(body) or ([body] if body else [])
        for sentence in sentences:
            if len(sentence) < MIN_SPAN_CHARS:
                continue
            matches = [(role, turn_index) for role, turn_index, source in sources if _span_survives(sentence, source)]
            if matches:
                role, turn_index = matches[0]
            elif sources:
                # Unsupported: attribute it to the first cited turn so the report names a real citation.
                role, turn_index = (sources[0][0], sources[0][1])
            else:
                continue
            span = LessonSpan(
                lesson_index=lesson_index,
                role=role,
                turn_index=turn_index,
                span=sentence,
                supported=bool(matches),
            )
            audit.spans.append(span)
            if matches:
                record = by_key.get(("user" if role == "U" else "assistant", turn_index))
                if record is not None:
                    record.matched_spans.append(sentence)

    for record in audit.turns:
        if record.role == "assistant" and not record.matched_spans:
            audit.notes.append(
                f"assistant turn {record.index}: no lesson quoted it, its content is unrepresented"
            )

    return audit


def lesson_texts(documents: Iterable[Any]) -> list[str]:
    """Flatten distill documents into plain lesson bodies.

    Upstream hands back ``DistillDocument`` objects with a ``body`` field and older releases return
    plain strings, so both shapes are accepted.
    """
    texts: list[str] = []
    for document in documents:
        if isinstance(document, str):
            texts.append(document)
            continue
        body = getattr(document, "body", None)
        texts.append(str(body) if body is not None else str(document))
    return texts
