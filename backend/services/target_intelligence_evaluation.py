"""Deterministic evaluation metrics for source-backed Target Intelligence cases.

This module evaluates supplied benchmark cases only. It never creates gold
targets, citations, or scientific labels.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping


UNIPROT_PATTERN = re.compile(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$", re.IGNORECASE)


def _normalized_ids(values: Iterable[Any]) -> list[str]:
    return [str(value).strip().upper() for value in values if isinstance(value, str) and str(value).strip()]


def _normalized_source_ids(values: Iterable[Any]) -> list[str]:
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        item = value.strip().upper()
        if item.isdigit():
            item = f"PMID:{item}"
        normalized.append(item)
    return normalized


def evaluate_target_ranking(predicted_ids: Iterable[Any], gold_ids: Iterable[Any], *, k: int = 10) -> dict[str, float | int | None]:
    """Evaluate one ranked target list against a curator-supplied gold set."""
    if k < 1:
        raise ValueError("k must be positive")
    predicted = _normalized_ids(predicted_ids)
    gold = set(_normalized_ids(gold_ids))
    top = predicted[:k]
    hits = [identifier for identifier in top if identifier in gold]
    first_rank = next((index + 1 for index, identifier in enumerate(predicted) if identifier in gold), None)
    dcg = sum(1.0 / math.log2(index + 2) for index, identifier in enumerate(top) if identifier in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return {
        "k": k,
        "predicted_count": len(top),
        "gold_count": len(gold),
        "hit_count": len(set(hits)),
        "precision_at_k": len(set(hits)) / len(top) if top else 0.0,
        "recall_at_k": len(set(hits)) / len(gold) if gold else None,
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "ndcg_at_k": dcg / idcg if idcg else None,
    }


def evaluate_identifier_normalization(predicted_ids: Iterable[Any], gold_ids: Iterable[Any]) -> dict[str, float | int]:
    """Measure exact reviewed-UniProt identifier agreement without fuzzy merging."""
    predicted = set(_normalized_ids(predicted_ids))
    gold = set(_normalized_ids(gold_ids))
    valid_predicted = {identifier for identifier in predicted if UNIPROT_PATTERN.fullmatch(identifier)}
    true_positive = len(valid_predicted & gold)
    precision = true_positive / len(valid_predicted) if valid_predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "valid_predicted_count": len(valid_predicted),
        "gold_count": len(gold),
        "true_positive_count": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_citations(citations: Iterable[Any], gold_source_ids: Iterable[Any]) -> dict[str, float | int]:
    """Measure exact citation-ID correctness against supplied source IDs."""
    predicted = _normalized_source_ids(citations)
    gold = set(_normalized_source_ids(gold_source_ids))
    correct = sum(1 for citation in predicted if citation in gold)
    return {
        "citation_count": len(predicted),
        "correct_count": correct,
        "correctness": correct / len(predicted) if predicted else 0.0,
    }


def evaluate_unsupported_claims(claims: Iterable[Mapping[str, Any]]) -> dict[str, float | int | None]:
    """Compute unsupported-claim rate from an audited claim list.

    Each claim must carry a curator/auditor-provided boolean ``supported``.
    Missing audit labels are not silently treated as supported.
    """
    claims = list(claims)
    audited = [claim for claim in claims if isinstance(claim, Mapping) and isinstance(claim.get("supported"), bool)]
    if not audited:
        return {"claim_count": len(claims), "audited_count": 0, "unsupported_count": 0, "unsupported_rate": None}
    unsupported = sum(1 for claim in audited if not claim["supported"])
    return {
        "claim_count": len(claims),
        "audited_count": len(audited),
        "unsupported_count": unsupported,
        "unsupported_rate": unsupported / len(audited),
    }
