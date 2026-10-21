"""Guardrails - the enterprise safety layer. BUILD_PLAN: Thread D.

check_citations enforces 'cite or drop': any claim the Critic did not confirm, or
that carries no evidence, is flagged and kept out of the trusted summary.
"""
from __future__ import annotations

from .models import Report


def check_citations(report: Report) -> Report:
    valid_ids = {e.source_id for e in report.sources}
    kept = []
    for c in report.claims:
        all_ids_real = bool(c.evidence_ids) and all(eid in valid_ids for eid in c.evidence_ids)
        if c.supported and all_ids_real:
            kept.append(c)
        else:
            if not c.evidence_ids:
                reason = "dropped: no citations"
            elif not all_ids_real:
                bad = [eid for eid in c.evidence_ids if eid not in valid_ids]
                reason = f"dropped: evidence_ids not in sources: {bad}"
            else:
                reason = "dropped: unsupported"
            c.note = (c.note + " | " + reason).strip(" |")
            report.unanswered.append(c.text)
    report.claims = kept
    return report
