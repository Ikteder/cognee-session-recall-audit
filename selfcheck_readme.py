"""Smoke check: the README example runs against stand-ins, no cognee needed.

Kept as a script rather than a test so the published example stays honest without making the suite
depend on the README's wording.
"""

from __future__ import annotations

import asyncio
from typing import Any

from audit import AuditReport, audit_curated_lessons, format_report, measure_session_lift
from audit.corpus import CURATED_LESSONS, SESSION_TURNS


class FakeRecall:
    """Returns a session-independent context unless a session id is supplied."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, query_text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        assert kwargs.get("only_context") is True, "the probe must stay LLM-free"
        context = f"graph answer for {query_text}"
        if kwargs.get("session_id"):
            context += " plus session note about the same incident"
        return {"answer": "", "context": context}


async def main() -> None:
    recall = FakeRecall()
    lift = await measure_session_lift(
        recall=recall,
        queries=["how do I rotate the staging deploy key", "restart the docker daemon"],
        session_id="incident-4417",
    )
    turns = [(role, content) for role, content in SESSION_TURNS]
    distillation = audit_curated_lessons(lessons=CURATED_LESSONS, turns=turns)
    report = AuditReport(lift=lift, distillation=distillation)
    print(format_report(report))

    assert recall.calls == 4, recall.calls
    assert len(lift) == 2
    assert all(item.lifted for item in lift), [item.query for item in lift]
    assert report.ok, report.probe_errors
    print("readme smoke: all assertions passed")


if __name__ == "__main__":
    asyncio.run(main())
