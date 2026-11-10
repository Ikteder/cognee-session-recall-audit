"""Fixture corpus: a small ops knowledge base plus one session that leans on it.

Everything here is fixed text so the audit numbers are reproducible across machines and runs. The
corpus is deliberately tiny; an audit kit does not need a large dataset to detect regressions, it
needs stable content whose expected retrieval behavior is known.
"""

from __future__ import annotations

# Dataset name used by every probe unless overridden on the command line.
DATASET_NAME = "session_recall_audit"

# Session id used by the probes. Reused across runs so a session's effect on retrieval is measured
# against the same graph, not a fresh one each time.
SESSION_ID = "audit-session-001"

# Documents ingested into the knowledge graph. Each one states a single operational rule with a
# distinctive term, which is what makes an expected-retrieval assertion meaningful.
DOCUMENT_TEXTS: tuple[str, ...] = (
    "Pod stuck in CrashLoopBackOff: check the container exit code with kubectl describe pod, then "
    "read the previous logs. A repeated 137 means an OOMKill; raise the memory limit and restart "
    "the deployment.",
    "Postgres connection pool exhaustion shows as too many connections. Increase the pool only "
    "after checking for leaked transactions with pg_stat_activity, otherwise the leak just returns.",
    "Certificate expiry on an ingress: renew with cert-manager, then restart the ingress "
    "controller so the new chain is loaded. Restarting first hides which certificate was stale.",
    "S3 upload failures with a 403 usually mean the bucket policy changed. Diff the policy against "
    "the last known good version before rotating credentials.",
    "Slow gRPC calls between services: measure per-hop latency with the mesh sidecar dashboard "
    "before tuning timeouts, because a timeout change masks the real hot path.",
)

# Turns of a session that queried the corpus. Roles follow cognee's QA turn shape (user then
# assistant). These are the inputs to the distillation coverage audit.
SESSION_TURNS: tuple[tuple[str, str], ...] = (
    (
        "user",
        "The checkout pod keeps restarting. How do I fix it?",
    ),
    (
        "assistant",
        "Check the container exit code with kubectl describe pod, then read the previous logs. "
        "A repeated 137 means an OOMKill, so raise the memory limit and restart the deployment.",
    ),
    (
        "user",
        "Ingress traffic started failing after the weekend. What changed?",
    ),
    (
        "assistant",
        "Certificate expiry on the ingress is the usual cause. Renew with cert-manager, then "
        "restart the ingress controller so the new chain is loaded.",
    ),
)

# A lesson set that a faithful curator would produce from SESSION_TURNS. Each entry is the
# ``working_statement`` of one ProposedLesson: a standalone sentence drawn from the turn it cites,
# plus the citation markers the audit reads. One quotes its source verbatim and the other compresses
# it, so this positive control exercises both support paths (exact containment and content-word
# traceability). Coverage must come out at 1.0 and nothing may be flagged.
CURATED_LESSONS: tuple[str, ...] = (
    "Check the container exit code with kubectl describe pod, then read the previous logs. [U1][A1]",
    "Raise the memory limit and restart the deployment. [U1][A1]",
    "Renew with cert-manager, then restart the ingress controller so the new chain is loaded. [U2][A2]",
)

# Queries probed for session lift. Each one is answerable from the graph, and at least one should
# come back different once the session layer contributes distilled lessons.
PROBE_QUERIES: tuple[str, ...] = (
    "The checkout pod keeps restarting. How do I fix it?",
    "Ingress traffic started failing after the weekend. What changed?",
    "How do I fix a repeated 137 exit code?",
)
