# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Audit the live cognee install: real recall, real session store, no LLM generation.

Requires a working cognee configuration (database plus embedding provider) and network access if
the corpus is ingested here. The measurement itself stays LLM free because every leg runs with
``only_context=True``.

Usage::

    python tools/audit_live.py                 # ingest the fixture corpus, then probe
    python tools/audit_live.py --skip-ingest   # dataset already built
    python tools/audit_live.py --session-id my-session --top-k 20

Exit status: 0 when the run is clean, 2 when cognee is unavailable or the live leg could not run
(missing database, bad embedding config). A measurement that comes back with zero lift exits 0; a
probe that never reached the session store does not, because those mean different things.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow "python tools/audit_live.py" from the repo root
    sys.path.insert(0, str(ROOT))

from audit.corpus import (
    CURATED_LESSONS,
    DATASET_NAME,
    DOCUMENT_TEXTS,
    PROBE_QUERIES,
    SESSION_ID,
)
from audit.distill_audit import audit_curated_lessons
from audit.recall_probe import measure_session_lift
from audit.report import AuditReport, format_report


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments. ``argv`` excludes the script name (pass ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(description="Measure cognee session memory against real recall.")
    parser.add_argument("--session-id", default=SESSION_ID)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--skip-ingest", action="store_true", help="assume the dataset is already built")
    parser.add_argument(
        "--distill",
        action="store_true",
        help="run cognee distillation instead of auditing the fixture lessons (makes LLM calls)",
    )
    parser.add_argument("--json", type=Path, help="also write the machine readable report here")
    return parser.parse_args(argv)


async def ingest(args: argparse.Namespace) -> None:
    import cognee

    print(f"ingesting {len(DOCUMENT_TEXTS)} fixture documents into '{args.dataset}'")
    for text in DOCUMENT_TEXTS:
        await cognee.remember(text, dataset_name=args.dataset)


def _lesson_texts(result) -> list[str]:
    documents = getattr(result, "documents", None)
    if not documents:
        return []
    return [str(document) for document in documents]


async def run(args: argparse.Namespace) -> int:
    import cognee
    from cognee import __version__ as cognee_version

    if not args.skip_ingest:
        await ingest(args)

    comparisons = await measure_session_lift(
        cognee.recall,
        PROBE_QUERIES,
        datasets=(args.dataset,),
        session_id=args.session_id,
        top_k=args.top_k,
    )

    distillation = None
    if args.distill:
        from cognee.modules.session_distillation import distill_session

        result = await distill_session(session_id=args.session_id, dataset=args.dataset)
        print(f"distillation status: {getattr(result, 'status', 'unknown')}")
        lessons = _lesson_texts(result)
        if lessons:
            from audit.corpus import SESSION_TURNS

            distillation = audit_curated_lessons(lessons, SESSION_TURNS)
    else:
        print(
            "distillation: auditing the fixture lesson set. Pass --distill to run real curation "
            "(that path makes LLM calls and is opt in)."
        )
        from audit.corpus import SESSION_TURNS

        distillation = audit_curated_lessons(CURATED_LESSONS, SESSION_TURNS)

    report = AuditReport(
        lift=comparisons,
        distillation=distillation,
        cognee_version=cognee_version,
    )
    print(format_report(report))

    if args.json:
        args.json.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    failed = [item for item in comparisons if item.error]
    if failed:
        print(f"{len(failed)} probe(s) never reached the session store; see the error rows above")
    return 1 if failed or not distillation.lessons else 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except ImportError as error:
        print(f"cognee is not importable in this environment: {error}", file=sys.stderr)
        print("run tools/audit_skill_gate.py and the offline suite instead", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - a broken database config should read as an error, not a traceback
        print(f"live audit failed: {type(error).__name__}: {error}", file=sys.stderr)
        print("this leg needs a working cognee configuration (database + embedding provider)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
