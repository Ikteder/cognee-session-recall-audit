# Cognee Session Recall Audit

A dependency-free audit kit for [cognee](https://github.com/topoteretes/cognee) session memory. It
answers the two questions every agent-memory deployment eventually has to answer with a number
instead of a demo:

> Did the memory layer actually change what the agent retrieved? And did distillation keep the
> lesson without inventing anything?

Both probes are deterministic and run **offline: no API key, no model download, no database**. The
full suite runs on a bare Python install in about a second.

![Audit suite](https://github.com/Ikteder/cognee-session-recall-audit/actions/workflows/audit.yml/badge.svg)

## At a glance

| | |
| --- | --- |
| Runtime dependencies | none (stdlib only) |
| LLM or embedding calls | zero, every probe uses `only_context=True` |
| Test time | ~0.03s offline (`pytest`: 36 passed, 1 skipped) |
| Python | 3.10 to 3.14, verified in CI across all five |
| Install | `pip install .` (wheel plus sdist), then `from audit import measure_session_lift` |
| Needs a cognee checkout | only for the live measurement legs, not for the suite |

## The gap this fills

cognee ships session memory with examples that demonstrate it end to end (`examples/guides/sessions.py`,
`session_distillation.py`). Every one of those needs an LLM key and a live database, and each prints
a verdict for a human to read. Nothing in the repo lets CI check the two claims that actually matter:

1. **Session lift.** Does passing `session_id` change what recall returns, or does the graph answer
   come back identical while the session layer sits there as decoration?
2. **Distillation coverage.** When a session is distilled into durable memory, which curated lessons
   survive into retrievable text, and which get dropped or reworded beyond recognition?

This kit turns both into numeric, assertable measurements with no LLM in the loop, so a memory
regression becomes a failing test rather than a subtly worse demo.

## What is here

```
audit/
  recall_probe.py    Session-lift probe: recall() with only_context=True, zero-LLM path
  distill_audit.py   Distillation coverage audit: curated lessons vs verbatim source turns
  corpus.py          Deterministic fixture corpus (ops runbook chunks, session turns, lessons)
  report.py          Structured results, verdict rules, plain-text table
selfcheck.py         Standalone verification of the measurement code, no cognee needed
selfcheck_readme.py  Runs the README example against stand-ins so the docs stay honest
skill_gate_ref.py    Pinned copy of the upstream skill gate rule set
tools/
  skill_gate_cases.json   Query matrix for the procedural skill gate (17 rows)
  audit_skill_gate.py     Runs the matrix against real cognee or the pinned copy
  audit_live.py           Runs the probes against a real cognee install
tests/
  test_offline_probes.py    Probes, fixtures and verdict rules
  test_skill_gate_matrix.py The matrix as a pytest suite
.github/workflows/audit.yml  CI for the offline legs plus a package build check
pyproject.toml
LICENSE                      MIT, with an explicit note about the vendored rule set
```

## Install

```bash
git clone https://github.com/Ikteder/cognee-session-recall-audit.git
cd cognee-session-recall-audit
pip install .            # installs the audit package only; still no dependencies
pip install ".[test]"    # adds pytest if you want to run the suite
```

Nothing in this kit pulls in cognee, an LLM provider SDK, or a model. The `audit` package is pure
stdlib, and cognee is reached through public interfaces at call time when you point the probes at it.

## Quickstart

Three entry points, from "no dependencies at all" to "measuring a real install".

```bash
python -m selfcheck        # verify the kit itself, no cognee required
python -m pytest -q        # 36 passed, 1 skipped, offline
python tools/audit_skill_gate.py   # skill gate matrix, exits 0 when only labeled issues remain
```

### 1. Verify the kit itself (no cognee, no network)

```bash
python -m selfcheck     # or: python selfcheck.py
```

The probes are written against cognee's documented API shapes. `selfcheck.py` feeds them stand-in
recall and curation callables that reproduce those shapes and asserts the exact numbers the probes
must produce (2 lifted queries out of 3, 100% assistant-turn coverage, 1 unsupported span caught).
This is what makes the measurement code verifiable on a machine that has never seen cognee.

### 2. Run the test suite

```bash
pip install pytest
python -m pytest -q
```

Expected: `36 passed, 1 skipped`. The skip is the guard that only runs when a real cognee install is
importable (see *Two gates, kept separate* below).

### 3. Measure a real cognee install

```bash
git clone https://github.com/topoteretes/cognee.git && cd cognee
uv sync --dev --all-extras --reinstall
cd /path/to/this/repo
python tools/audit_skill_gate.py          # uses the installed gate automatically
python tools/audit_live.py                # needs a configured database and embedding provider
```

`audit_skill_gate.py` prints one row per query and exits 1 if any behavior moved without a known
issue attached. `audit_live.py` ingests the fixture corpus and measures real `recall()`; that leg
does need a working cognee configuration (database plus embedding provider), but it still makes no
LLM calls, because every probe runs with `only_context=True`.

## How the probes work

### 1. Session lift (`audit/recall_probe.py`)

Runs `cognee.recall(query_text=..., only_context=True)` twice for the same query, once without a
session and once with `session_id`, then compares the assembled retrieval context:

* identical contexts mean the session layer contributed nothing for that query;
* different content means it did, and the delta is reported in characters;
* a session leg that raises is recorded as an **error**, not as "no change".

`only_context=True` returns the prompt an LLM *would* have received instead of an answer. That keeps
the probe LLM-free and, more importantly, measures the part memory actually controls. Comparing
answer strings would fold ordinary sampling noise into a memory signal.

Three buckets come out of every run: `lifted`, `unchanged`, `errored`. They are deliberately not two
buckets. A harness that failed to reach the session store must never be readable as evidence that
memory adds nothing, so an errored row can never count as lift and fails the report verdict.

### 2. Distillation coverage (`audit/distill_audit.py`)

Distillation curates session turns into lessons and then writes them through an LLM writer. That
writer is where content gets dropped or invented. This probe audits the curated lesson set directly:

* every lesson carries `[U#]` / `[A#]` citations; each span must trace back to the turn it cites,
  either verbatim after normalization or through content-word traceability (a compressed lesson that
  reorders words is supported, a lesson containing claims the source never made is not);
* each assistant turn is scored for how much of its own text survived into some lesson;
* any span that cannot be traced is listed as unsupported, and any turn that contributed nothing is
  listed as uncovered.

User turns are **not** scored. Upstream curation explicitly rejects breadcrumbs like "User asked about
X", so an unquoted question is normal and counting it would produce noise on every faithful curator.
Only assistant turns can be lost content. The report prints them as `n/a` rather than as failures.

### 3. Skill gate matrix (`tools/audit_skill_gate.py`)

The procedural skill gate in `cognee/api/v1/recall/skill_gate.py` decides whether a query also gets a
runbook lookup. It is weighted regex with negation suppression, which drifts silently: one edited
character class changes behavior for thousands of queries and no test notices. The JSON matrix pins
the expected fired/not-fired outcome for 17 English and Chinese queries.

Rows carry an optional `known_failure` label. A row that deviates from its expectation is reported as
`known` when a labeled issue covers it and `unexpected` otherwise. Only `unexpected` fails the run, so
the audit stays green on a clean checkout and turns red the moment upstream behavior moves for an
unexplained reason.

## Two gates, kept separate

The kit can measure two different things, and conflating them is how an audit quietly stops meaning
anything:

| Gate | What it validates | Requires |
| --- | --- | --- |
| `skill_gate_ref.py` (pinned copy) | that the expectations file still describes the rule set as vendored | nothing |
| `tools/audit_skill_gate.py` default | that the **installed** cognee still behaves as pinned | a cognee checkout |

By default the tool resolves to the installed gate and prints which one it used. If cognee is not
importable it falls back to the reference copy and says so loudly, because in that mode it is testing
the kit's own expectations, not shipped behavior. `test_reference_copy_stays_in_sync_with_the_matrix`
catches drift between the vendored rules and the matrix, and a skipif-guarded test asserts that an
upstream run really did load the installed gate instead of falling back.

The vendored copy is ASCII-only in its clause splitter on purpose, mirroring upstream. Widening it to
CJK punctuation would locally "fix" the open bug below and stop the matrix from reproducing it, so the
constraint is documented inline where the regex lives rather than left implicit.

## Known findings

Running the matrix against cognee `main` reproduces an open upstream bug:

* [#5647](https://github.com/topoteretes/cognee/issues/5647): a negation in one clause cancels a
  procedural match in a later one, so queries like "Docker won't start, how do I fix it?" never get a
  runbook lookup. The gate is meant to suppress "don't walk me through it", not every query that
  mentions something broken.

The matrix separates what upstream gets right today from what it still gets wrong. English clause
breaks (`.`, `,`, `;`) are honored, so the five English negation-leak rows fire correctly and stay in
the file to pin that boundary. The Chinese set is where the leak is still live: upstream's clause split
covers ASCII punctuation only, so a full-width comma is not a boundary and `不能` is scoped to cancel
every match in the query. One row (`部署失败了，不能访问，怎么配置回滚？`) reproduces that leak and is
labeled `known_failure: 5647`. A test asserts the leaking count stays at exactly one, which pins both
directions: if upstream fixes the split the count drops and the suite says so, and if the reference
copy drifts toward CJK clause support it goes up and fails.

The matrix also pins the English/Chinese rule asymmetry added in #5428: Chinese procedural rules carry
weight 1 while English step-by-step phrases carry weight 2, which leaves several Chinese queries sitting
exactly on the firing threshold.

## Why this matters

Memory features usually ship on the strength of a demo that shows them working once. These probes give
a maintainer or an integrator three cheap offline checks that belong next to an existing unit suite:

* a regression in the session layer becomes a failing number instead of a subtly worse demo;
* distillation losses become an explicit list of unsupported spans instead of a silent omission;
* skill gate edits become a reviewed diff against a pinned matrix instead of an untested regex change.

They are small and deterministic on purpose: no fixtures to regenerate, no golden files to re-bless,
no provider quota to burn.

## Using it as a library

```python
import asyncio

import cognee
from audit import AuditReport, audit_curated_lessons, format_report, measure_session_lift

async def main() -> None:
    lift = await measure_session_lift(
        recall=cognee.recall,
        queries=["how do I rotate the staging deploy key", "restart the docker daemon"],
        session_id="incident-4417",
    )
    turns = [(message.role, message.content) for message in await cognee.session_get_messages("incident-4417")]
    distillation = audit_curated_lessons(lessons=curated_lessons, turns=turns)  # your curation output
    report = AuditReport(lift=lift, distillation=distillation)
    print(format_report(report))               # human readable table
    assert report.ok, report.probe_errors      # numeric verdict for CI

asyncio.run(main())
```

`measure_session_lift` is async and takes any callable matching `recall(query_text=..., **kwargs)`, so
it works with cognee today and with any other memory backend that can return a retrieval context. The
distillation audit is pure data in, data out: no calls at all.

## Requirements

Python 3.10 to 3.14. The offline path has **no runtime dependencies at all**; `pytest` is used by the
test files only. Measuring a real install needs a cognee checkout from source. Nothing in this kit
needs an LLM provider or an embedding model, and nothing calls a paid API.

## Relationship to upstream

This is an independent audit kit: not a fork, not a wrapper, and it contains no cognee source. It
reaches cognee through public interfaces (`cognee.recall`, the session distillation module,
`should_search_skills`) and adds measurement on top. The test files are written to drop into a cognee
checkout as regression tests without rework:

| This repository | Upstream target |
| --- | --- |
| `tests/test_offline_probes.py` (lift half) | `cognee/tests/unit/api/v1/recall/test_session_recall_probe.py` |
| `tests/test_offline_probes.py` (coverage half) | `cognee/tests/unit/modules/session_distillation/test_distill_coverage_audit.py` |
| `tests/test_skill_gate_matrix.py` + `tools/skill_gate_cases.json` | `cognee/tests/unit/api/v1/recall/` |

The probes also work standalone: point them at any cognee install and they measure it, which makes them
usable as an integration check for anyone self-hosting session memory.

## License

MIT. Cognee itself is Apache 2.0; this kit imports its public interfaces only. `skill_gate_ref.py` is a
short verbatim copy of one upstream rule set, kept under the same terms as cognee and marked with the
version it mirrors, so anyone can diff it against the original. The LICENSE file records that split.

Cognee (topoteretes/cognee) is not affiliated with this kit and does not endorse it.
