"""Local copy of cognee's procedural skill gate, for environments without cognee installed.

Source: https://github.com/topoteretes/cognee/blob/main/cognee/api/v1/recall/skill_gate.py
The matrix test prefers the real module whenever one is importable and falls back to this file so
the audit kit still measures something on a bare machine. Keep this copy in sync by hand; the
version line below records what it mirrors.

Mirrored: cognee main branch, skill gate as of 2026-11 (procedural rule set with same-clause
negation suppression).

Copyright (c) 2026 Ikteder. Portions derived from topoteretes/cognee, Copyright the cognee authors,
both licensed under the Apache License, Version 2.0. See LICENSE at the repository root.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SkillGateResult:
    fired: bool
    score: float
    matched: tuple[str, ...] = ()


# Procedural intent signals: (pattern, weight). A cumulative score >= 1.0 triggers lookup.
_PROCEDURAL_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    # English explicit procedural phrases
    (re.compile(r"\b(?:how|what|where|when|which)\s+(?:do|does|did|can|could|should|would|will|is|are|was|were)\s+i\b", re.I), 1.0),
    (re.compile(r"\bi\s+(?:need|want|have)\s+to\s+(?:know|find|learn|figure|understand)", re.I), 1.0),
    (re.compile(r"\bhow\s+(?:can|do|could|should|would|to)\s+(?:i|we|you)\b", re.I), 1.0),
    (re.compile(r"\bhow\s+(?:to|should|can|do|does|did)\b", re.I), 1.0),
    (re.compile(r"\bwhat(?:'s| is| are| would| should| can)\b", re.I), 0.5),
    (re.compile(r"\b(?:step[- ]by[- ]step|walk\s+me\s+through|show\s+me|guide\s+me|teach\s+me)", re.I), 2.0),
    (re.compile(r"\bi'?m\s+(?:trying|looking)\s+to\b", re.I), 1.5),
    (re.compile(r"\b(?:fix|solve|resolve|troubleshoot|debug|set\s?up|install|configure|deploy|migrate|implement|write|create|build)\b", re.I), 0.5),
    (re.compile(r"\?\s*$"), 0.5),
    # Chinese procedural intent signals
    (re.compile(r"[?]？"), 0.5),
    (re.compile(r"怎么|怎样|如何|哪里|为什么|什么时候|哪个"), 1.0),
    (re.compile(r"(?:需要|要|想|必须)\s*(?:知道|了解|找到|学习|弄明白|搞清楚)"), 1.0),
    (re.compile(r"分步|逐步|一步步|教(我|一下)|演示|展示|指导|带我"), 2.0),
    (re.compile(r"(?:帮忙|帮助|协助)(?:我)?"), 1.5),
]

# Negation markers that suppress procedural intent: (pattern, scope). Scope is "clause" when the
# negation only cancels matches in the same clause, or "all" when it always cancels every match.
_NEGATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # English contractions and full forms (clause-scoped)
    (re.compile(r"\b(?:don'?t|doesn'?t|didn'?t|can'?t|cannot|couldn'?t|shouldn'?t|won'?t|wouldn'?t|isn'?t|aren'?t|wasn'?t|weren'?t|haven'?t|hasn'?t|hadn'?t|am\s+not|not\s+(?:going|able|allowed|supposed))\b", re.I), "clause"),
    (re.compile(r"\bno\s+idea\b", re.I), "clause"),
    # Chinese negation and inability markers (clause-scoped)
    (re.compile(r"不[能要会想](?:\s*(?:知道|了解|找到|学习|弄明白|搞清楚))?"), "clause"),
    (re.compile(r"别(?!名|人|处|的)"), "clause"),
    (re.compile(r"(?:无需|不用|不需要|没必要|不要|不想|不会|不能|不可以|没(?=有)|无法)"), "all"),
]

# Clause boundaries: sentence terminators and clause-joining punctuation. A clause-scoped negation
# cancels procedural matches only within the same clause, so a negation in an earlier clause does not
# suppress a procedural match in a later one.
#
# The ASCII ranges below are deliberate. Upstream splits on ``[.!?,;:…]`` plus the em/en dash pair and
# nothing else, so CJK ideographic commas and full stops (U+3002, U+FF0C, U+FF1B) are NOT boundaries.
# A Chinese negation therefore leaks across clauses and cancels procedural signals in later clauses,
# which is the behavior tracked upstream as cognee #5647. Widening this set would silently "fix" the
# bug locally and stop the matrix from reproducing it, so the reference copy stays ASCII-only and any
# change to these ranges must be mirrored upstream first.
_CLAUSE_SPLIT = re.compile(r"[.!?,;:\u2014\u2015]")


def should_search_skills(query_text: str) -> SkillGateResult:
    """Return whether a query looks procedural enough to justify a skill lookup.

    A cumulative score >= 1.0 is required, so no single weak signal (weight 0.5) fires alone; two
    weak signals or one strong one suffice. Negation markers cancel matches within the same clause
    as the negation, while ``no idea`` and the Chinese negation set cancel all matches anywhere.

    Known limitation tracked upstream (cognee #5647): the clause split is ASCII punctuation only, so
    Chinese clause breaks are not honored and a Chinese negation in one clause suppresses procedural
    signals in later clauses too.
    """
    if not query_text or not query_text.strip():
        return SkillGateResult(fired=False, score=0.0)

    text = query_text.lower()
    clauses = [c for c in _CLAUSE_SPLIT.split(text) if c.strip()]

    matched: list[str] = []
    negated_anywhere = False
    suppressed_clauses: set[int] = set()

    for pattern, scope in _NEGATION_PATTERNS:
        if scope == "all" and pattern.search(text):
            negated_anywhere = True
        elif scope == "clause":
            for index, clause in enumerate(clauses):
                if pattern.search(clause):
                    suppressed_clauses.add(index)

    if negated_anywhere:
        return SkillGateResult(fired=False, score=0.0, matched=())

    score = 0.0
    for pattern, weight in _PROCEDURAL_PATTERNS:
        hits_in_unsuppressed = sum(
            len(pattern.findall(clause))
            for index, clause in enumerate(clauses)
            if index not in suppressed_clauses
        )
        if hits_in_unsuppressed:
            score += weight * min(hits_in_unsuppressed, 2)
            matched.append(pattern.pattern)

    return SkillGateResult(fired=score >= 1.0, score=score, matched=tuple(matched))
