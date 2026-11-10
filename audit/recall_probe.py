"""Session-lift probe: measure what cognee's session layer adds to retrieval.

The probe calls the real :func:`cognee.recall` with ``only_context=True``. That mode returns the
prompt an LLM *would* have received instead of an answer, so no language model runs and the
measurement is about retrieved content rather than model phrasing.

Why compare contexts and not answers: two identical retrievals can produce different answers, and
a memory regression looks exactly like ordinary sampling noise. The retrieval context is the part
the memory layer controls, so it is what gets compared.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from audit.corpus import DATASET_NAME, SESSION_ID


@dataclass
class RecallComparison:
    """Result of running one query with and without a session.

    Attributes:
        query: The query text that was probed.
        graph_context: Assembled retrieval context with no session in play.
        session_context: Assembled retrieval context with the session layer available.
        search_types: Search types reported by each run, for provenance of the comparison.
    """

    query: str
    graph_context: str = ""
    session_context: str = ""
    search_types: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def identical(self) -> bool:
        """True when the session layer changed nothing in the retrieval context."""
        return self.graph_context == self.session_context

    @property
    def delta_chars(self) -> int:
        """Signed size change of the session run relative to the graph-only run."""
        return len(self.session_context) - len(self.graph_context)

    @property
    def lifted(self) -> bool:
        """True when the session layer demonstrably contributed content to the retrieval context.

        A failed or skipped session leg leaves ``session_context`` empty. That is an unknown, not a
        contribution, so it must never count as lift.
        """
        return bool(self.session_context) and not self.identical

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "identical": self.identical,
            "lifted": self.lifted,
            "delta_chars": self.delta_chars,
            "graph_chars": len(self.graph_context),
            "session_chars": len(self.session_context),
            "search_types": list(self.search_types),
            "notes": list(self.notes),
            **({"error": self.error} if self.error else {}),
        }


def extract_context(result: Any) -> str:
    """Pull the retrieval context out of a recall result.

    ``recall()`` returns ``source``-tagged entries whose ``text`` field holds either an answer or,
    with ``only_context=True``, the prompt content. Entries may be objects or plain dicts depending
    on the cognee release, so both shapes are read.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result

    entries: Sequence[Any]
    if isinstance(result, dict):
        entries = [result]
    elif isinstance(result, (list, tuple)):
        entries = list(result)
    else:
        entries = [result]

    parts: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            text = entry.get("text") or entry.get("context") or entry.get("content")
        else:
            text = getattr(entry, "text", None)
            if text is None:
                text = getattr(entry, "context", None)
        if text is None:
            text = str(entry)
        parts.append(str(text))
    return "\n".join(parts)


def extract_search_types(result: Any) -> tuple[str, ...]:
    """Collect the ``search_type`` each graph entry reports, in order."""
    if not isinstance(result, (list, tuple)):
        return ()
    found: list[str] = []
    for entry in result:
        value = entry.get("search_type") if isinstance(entry, dict) else getattr(entry, "search_type", None)
        if value is not None:
            found.append(str(value))
    return tuple(found)


async def compare_recall_context(
    recall: Callable[..., Any],
    query: str,
    *,
    datasets: Sequence[str] = (DATASET_NAME,),
    session_id: str | None = SESSION_ID,
    top_k: int = 15,
    extra_kwargs: dict[str, Any] | None = None,
) -> RecallComparison:
    """Run ``query`` twice through ``recall`` and compare the retrieval contexts.

    Args:
        recall: The callable to use, normally :func:`cognee.recall`. Injected so the probe is
            testable without a database and so callers can point it at a pinned API surface.
        query: Query text to probe.
        datasets: Dataset names to search. Passing explicit datasets keeps the probe out of
            cognee's dataset routing path, which would otherwise add a variable number of datasets.
        session_id: Session used for the second run. ``None`` disables the session leg.
        top_k: Retrieval depth, pinned so both legs retrieve the same number of rows.
        extra_kwargs: Passed through to ``recall`` unchanged (for example ``query_type=``).

    Returns:
        A :class:`RecallComparison` describing what the session layer contributed.
    """
    extra = dict(extra_kwargs or {})

    async def _run(**kwargs: Any) -> Any:
        call = recall(**kwargs)
        if inspect.isawaitable(call):
            call = await call
        return call

    base_kwargs: dict[str, Any] = {
        "query_text": query,
        "datasets": list(datasets),
        "top_k": top_k,
        "only_context": True,
        **extra,
    }

    graph_result = await _run(**base_kwargs)
    comparison = RecallComparison(
        query=query,
        graph_context=extract_context(graph_result),
        search_types=extract_search_types(graph_result),
    )

    if session_id is None:
        comparison.notes.append("session leg skipped (session_id=None)")
        return comparison

    try:
        session_result = await _run(**base_kwargs, session_id=session_id)
    except Exception as error:  # noqa: BLE001 - the probe reports, it does not swallow silently
        comparison.error = f"{type(error).__name__}: {error}"
        comparison.notes.append(f"session leg failed: {comparison.error}")
        return comparison

    comparison.session_context = extract_context(session_result)
    comparison.search_types = comparison.search_types + extract_search_types(session_result)
    return comparison


async def measure_session_lift(
    recall: Callable[..., Any],
    queries: Sequence[str],
    **kwargs: Any,
) -> list[RecallComparison]:
    """Probe every query and return one comparison per query, in order."""
    return [await compare_recall_context(recall, query, **kwargs) for query in queries]
