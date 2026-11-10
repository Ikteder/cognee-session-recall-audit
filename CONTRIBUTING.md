# Contributing

Thanks for taking an interest in a memory audit kit. The whole point of this repository is that its
verdicts are reproducible, so the bar for a change is "the offline suite still says the same thing on
a clean machine".

## Ground rules

- **Keep the offline path dependency-free.** The `audit` package stays stdlib only. If a change needs a
  new import there, argue for it in the issue first.
- **No LLM calls in any probe or test.** Every measurement uses retrieval-only modes such as
  `recall(only_context=True)`. A test that needs an API key cannot be merged, because CI has none and
  results would not be reproducible.
- **Deterministic fixtures only.** Numbers asserted in tests come from `audit/corpus.py` or inline
  stand-ins, never from a live provider.
- **Distinguish measurement from expectation.** If upstream behavior moved, update the matrix with a
  `known_failure` label pointing at an actual issue, or say so in the PR. Do not silently re-bless
  output.

## Before opening a PR

```bash
python -m selfcheck          # measurement code, no cognee required
python -m selfcheck_readme   # keeps the README example honest
python -m pytest -q          # expect 36 passed, 1 skipped
python tools/audit_skill_gate.py --reference
python -m build && python -m twine check dist/*
```

All five must pass. CI runs the same set across Python 3.10 to 3.14.

## Reporting a problem

Include the exact command, the full output, your Python version, and whether cognee was importable at
the time. A report that says "the audit failed" without saying which leg failed is hard to act on: an
errored probe and a zero-lift measurement look similar in a screenshot but mean opposite things.

## Scope

This kit measures retrieval changes and lesson traceability. It is not a memory framework, not a cognee
fork, and not a benchmark suite for LLM quality. Features outside those two probes are more likely to be
pointed at an upstream issue than merged here.
