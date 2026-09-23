"""Deterministic query-intent routing for Target Intelligence."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


SUPPORTED = "SUPPORTED"
NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

DOMAIN_TERMS = {
    "disease", "diseases", "cancer", "syndrome", "disorder", "phenotype", "mechanism",
    "pathway", "protein", "proteins", "target", "targets", "gene", "genes", "receptor",
    "enzyme", "kinase", "transporter", "biomarker", "bioactivity", "chembl", "uniprot",
    "qsar", "drug", "compound", "inhibitor", "binding", "assay", "toxicity", "toxicology",
    "pharmacology", "therapeutic", "molecular", "cellular", "human", "mouse", "organism",
}

GENERIC_RESEARCH_TERMS = {
    "disease", "diseases", "phenotype", "mechanism", "pathway", "protein", "proteins",
    "target", "targets", "gene", "genes", "human", "organism", "drug", "compound",
    "compounds", "inhibitor", "inhibitors", "qsar", "screening", "bioactivity", "data",
    "use", "find", "identify", "recommend", "suggest", "which", "what", "should",
}

OUT_OF_SCOPE_TERMS = {
    "weather", "recipe", "football", "soccer", "movie", "song", "poem", "story", "joke",
    "password", "email", "invoice", "shopping", "travel", "flight", "translate", "translation",
    "javascript", "python code", "debug my code", "write code", "essay", "resume",
}


@dataclass(frozen=True)
class QueryIntent:
    status: str
    message: str
    normalized_question: str
    clarifying_questions: tuple[str, ...] = ()
    detected_concepts: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_query_intent(question: str) -> QueryIntent:
    normalized = " ".join(str(question or "").split()).strip()
    lowered = normalized.lower()
    tokens = set(re.findall(r"[a-z][a-z0-9-]{2,}", lowered))

    if len(normalized) < 3:
        return QueryIntent(
            NEEDS_CLARIFICATION,
            "Please describe a disease, phenotype, mechanism, gene, protein, or QSAR objective.",
            normalized,
            ("What biological problem are you investigating?", "Which target type or organism should be prioritized?"),
        )

    if any(term in lowered for term in OUT_OF_SCOPE_TERMS):
        return QueryIntent(
            OUT_OF_SCOPE,
            "This assistant supports literature-guided biological target discovery and QSAR planning only.",
            normalized,
            ("Provide a disease, phenotype, mechanism, gene, protein, or drug-discovery question.",),
        )

    detected = sorted(token for token in tokens if token in DOMAIN_TERMS)
    if detected:
        contextual_terms = tokens - GENERIC_RESEARCH_TERMS
        if not contextual_terms:
            return QueryIntent(
                NEEDS_CLARIFICATION,
                "I need a disease, phenotype, mechanism, or specific target context before recommending targets.",
                normalized,
                (
                    "Which disease, phenotype, or biological process is relevant?",
                    "Should I prioritize a protein target, gene, pathway, or mechanism?",
                    "Should the result emphasize biological evidence, QSAR readiness, or both?",
                ),
                tuple(detected),
            )
        return QueryIntent(
            SUPPORTED,
            "The question is suitable for literature-guided target discovery.",
            normalized,
            detected_concepts=tuple(detected),
        )

    if any(marker in lowered for marker in ("find", "identify", "recommend", "suggest", "which", "what")):
        return QueryIntent(
            NEEDS_CLARIFICATION,
            "I need a biological context before recommending targets.",
            normalized,
            (
                "Which disease, phenotype, or biological process is relevant?",
                "Should I prioritize a protein target, gene, pathway, or mechanism?",
                "Should the result emphasize biological evidence, QSAR readiness, or both?",
            ),
        )

    return QueryIntent(
        OUT_OF_SCOPE,
        "I could not identify a biological target-discovery question in this input.",
        normalized,
        ("Provide a disease, phenotype, mechanism, gene, protein, or QSAR question.",),
    )
