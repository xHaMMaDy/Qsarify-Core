"""Evidence-first literature and target retrieval for QSARify.

The retrieval and scoring core remains deterministic: it retrieves live records
from public APIs, preserves source identifiers, and owns target scores. An
optional LLM evidence-extraction layer is invoked separately and is never
allowed to overwrite deterministic identifiers or scores.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import requests

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_TYPES = ("europe_pmc", "pubmed", "chembl", "uniprot", "openalex", "crossref")
DEFAULT_WEIGHTS = {"biological_relevance": 0.4, "evidence_quality": 0.3, "qsar_readiness": 0.3}
USER_AGENT = "QSARify-Target-Intelligence/0.1 (research software)"


@lru_cache(maxsize=2048)
def _hgnc_canonical_symbol(token: str) -> str | None:
    """Resolve an exact human symbol/alias through HGNC when available."""
    token = str(token or "").strip().upper()
    if not token:
        return None
    try:
        payload = _request_json("https://rest.genenames.org/search", params={"q": f"symbol:{token} OR alias_symbol:{token}"}, timeout=10, retries=1)
        docs = ((payload.get("response") or {}).get("docs") or [])
        approved = [doc for doc in docs if str(doc.get("status") or "").lower() == "approved"]
        if approved:
            return str(approved[0].get("symbol") or "").upper() or None
    except TargetIntelligenceError:
        return None
    return None


class TargetIntelligenceError(RuntimeError):
    """Base error for bounded, source-aware retrieval failures."""


@dataclass(frozen=True)
class SourceRecord:
    source_type: str
    stable_id: str
    title: str | None = None
    url: str | None = None
    doi: str | None = None
    publication_year: int | None = None
    evidence_tier: str = "E"
    abstract_or_excerpt: str | None = None
    retrieved_at: str | None = None
    source_snapshot_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TargetCandidate:
    canonical_name: str
    uniprot_accession: str
    uniprot_reviewed: bool
    gene_symbols: list[str] = field(default_factory=list)
    organism: str = "Homo sapiens"
    protein_id: str | None = None
    chembl_target_id: str | None = None
    chembl_target_name: str | None = None
    chembl_activity_count: int | None = None
    chembl_status: str = "not_yet_checked"
    chembl_activity_status: str = "not_yet_checked"
    literature_hits: int = 0
    supporting_source_ids: list[str] = field(default_factory=list)
    conflicting_source_ids: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    evidence_role: str = "unclassified"


def normalize_weights(weights: Mapping[str, Any] | None) -> dict[str, float]:
    """Validate configurable composite weights without silently changing them."""
    raw = dict(DEFAULT_WEIGHTS if weights is None else weights)
    keys = tuple(DEFAULT_WEIGHTS)
    try:
        values = {key: float(raw[key]) for key in keys}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ranking_weights must contain biological_relevance, evidence_quality, and qsar_readiness") from exc
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("ranking_weights must contain finite non-negative values")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
        raise ValueError("ranking_weights must sum to 1")
    return values


def _bounded_text(value: Any, limit: int = 12000) -> str:
    return str(value or "").strip()[:limit]


def _snapshot_hash(record: Mapping[str, Any]) -> str:
    stable = repr(sorted((str(key), str(value)) for key, value in record.items()))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _year(value: Any) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    return int(match.group(0)) if match else None


def _safe_search_terms(question: str, limit: int = 300) -> str:
    """Reduce free text to bounded search terms before using source syntax."""
    # Natural-language questions contain ranking instructions that are useful
    # for literature retrieval but noisy for UniProt keyword search. Search for
    # a disease/phenotype phrase first across the whole question. The previous
    # implementation accepted the first generic relation phrase, so a question
    # ending in "for QSAR modeling" was searched as "QSAR modeling" rather
    # than as the disease named earlier in the sentence.
    # Comparison questions often name several human targets explicitly. Keep
    # only identifier-like uppercase tokens and exclude domain boilerplate so
    # the source query remains bounded and meaningful.
    excluded = {"AND", "ARE", "BUT", "FOR", "THE", "WITH", "HUMAN", "QSAR", "CHEMBL", "PMID", "DOI"}
    target_tokens = []
    for token in re.findall(r"\b[A-Z][A-Z0-9-]{1,9}\b", question):
        if token not in excluded and token not in target_tokens:
            target_tokens.append(token)
    if len(target_tokens) >= 2:
        return " OR ".join(target_tokens)[:limit]

    # Keep a compact disease/phenotype phrase for UniProt. The previous
    # pattern required a preposition and missed common forms such as
    # "Parkinson disease" and "multiple sclerosis", causing the entire
    # natural-language question to be sent as a keyword query.
    disease_suffix = (
        r"disease|cancer|syndrome|disorder|condition|leukemia|carcinoma|"
        r"fibrosis|infection|sclerosis|dementia|epilepsy|arthritis|"
        r"diabetes|asthma|melanoma|lymphoma|myeloma|stroke|lupus"
    )
    phrase_matches = list(re.finditer(
        rf"\b(?:[A-Za-z][A-Za-z0-9'-]*\s+){{0,3}}(?:{disease_suffix})\b",
        question,
        flags=re.IGNORECASE,
    ))
    if phrase_matches:
        candidate_text = phrase_matches[-1].group(0)
        ignored = {"which", "what", "that", "have", "has", "with", "for", "and", "the", "most", "strongly", "associated", "related", "relevant"}
        terms = [
            term for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", candidate_text)
            if term.lower() not in ignored
        ]
        return " ".join(terms)[:limit]

    # Last-resort bounded terms for non-disease questions. Do not preserve
    # punctuation or raw URL/query syntax in downstream source requests.
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", question)
    return " ".join(terms)[:limit]


def _request_json(url: str, *, params: Mapping[str, Any] | None = None, timeout: float = 15.0, retries: int = 2) -> dict[str, Any]:
    """Perform a real bounded JSON request with transient retry handling."""
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(min(2**attempt, 4))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TargetIntelligenceError(f"Unexpected non-object response from {url}")
            return payload
        except (requests.RequestException, ValueError, TargetIntelligenceError) as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(exc, requests.HTTPError) and status not in {429, 500, 502, 503, 504}:
                break
            if attempt < retries:
                time.sleep(min(2**attempt, 4))
    raise TargetIntelligenceError(f"Source request failed: {url}") from last_error


def _source(source_type: str, stable_id: str, *, title: Any = None, url: Any = None, doi: Any = None,
            year: Any = None, tier: str = "E", excerpt: Any = None, metadata: Mapping[str, Any] | None = None) -> SourceRecord:
    metadata = dict(metadata or {})
    return SourceRecord(
        source_type=source_type,
        stable_id=_bounded_text(stable_id, 300),
        title=_bounded_text(title, 1000) or None,
        url=_bounded_text(url, 2000) or None,
        doi=_bounded_text(doi, 300) or None,
        publication_year=_year(year),
        evidence_tier=tier,
        abstract_or_excerpt=_bounded_text(excerpt, 8000) or None,
        retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_snapshot_hash=_snapshot_hash(metadata),
        metadata=metadata,
    )


def search_europe_pmc(question: str, *, limit: int = 20) -> list[SourceRecord]:
    payload = _request_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": question, "format": "json", "resultType": "core", "pageSize": min(limit, 100)},
    )
    records: list[SourceRecord] = []
    for item in payload.get("resultList", {}).get("result", []):
        if not isinstance(item, dict):
            continue
        pmid = item.get("pmid")
        stable_id = f"PMID:{pmid}" if pmid else f"EPMC:{item.get('id', '')}"
        if stable_id.endswith(":"):
            continue
        pub_types = {str(value).lower() for value in item.get("pubTypeList", {}).get("pubType", [])}
        tier = "A" if any(term in pub_types for term in ("clinical trial", "randomized controlled trial", "journal article")) else "B"
        records.append(_source(
            "europe_pmc", stable_id, title=item.get("title"),
            url=f"https://europepmc.org/article/MED/{pmid}" if pmid else None,
            doi=item.get("doi"), year=item.get("firstPublicationDate") or item.get("pubYear"),
            tier=tier, excerpt=item.get("abstractText"),
            metadata={"authorString": item.get("authorString"), "pubType": list(pub_types), "pmcid": item.get("pmcid")},
        ))
    return records


def search_pubmed(question: str, *, limit: int = 20) -> list[SourceRecord]:
    search = _request_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": question, "retmode": "json", "retmax": min(limit, 100)},
    )
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    summary = _request_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
    )
    records: list[SourceRecord] = []
    for pmid in ids:
        item = summary.get("result", {}).get(str(pmid), {})
        if not isinstance(item, dict):
            continue
        records.append(_source(
            "pubmed", f"PMID:{pmid}", title=item.get("title"),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", year=item.get("pubdate"),
            tier="A", metadata={"source": item.get("source"), "fulljournalname": item.get("fulljournalname")},
        ))
    return records


def search_openalex(question: str, *, limit: int = 20) -> list[SourceRecord]:
    search_terms = _safe_search_terms(question)
    payload = _request_json(
        "https://api.openalex.org/works",
        params={"search": search_terms, "per-page": min(limit, 100), "select": "id,doi,title,publication_year,open_access"},
    )
    records: list[SourceRecord] = []
    for item in payload.get("results", []):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        records.append(_source(
            "openalex", str(item["id"]), title=item.get("title"),
            url=item.get("doi") or item.get("id"), doi=item.get("doi"),
            year=item.get("publication_year"), tier="C", metadata={"open_access": item.get("open_access")},
        ))
    return records


def search_crossref(question: str, *, limit: int = 20) -> list[SourceRecord]:
    payload = _request_json(
        "https://api.crossref.org/works",
        params={"query.bibliographic": question, "rows": min(limit, 100), "select": "DOI,title,published,URL,type"},
    )
    records: list[SourceRecord] = []
    for item in payload.get("message", {}).get("items", []):
        if not isinstance(item, dict) or not item.get("DOI"):
            continue
        title = (item.get("title") or [None])[0]
        date_parts = ((item.get("published") or {}).get("date-parts") or [[None]])[0]
        records.append(_source(
            "crossref", f"DOI:{item['DOI']}", title=title, url=item.get("URL"), doi=item.get("DOI"),
            year=date_parts[0] if date_parts else None, tier="C", metadata={"type": item.get("type")},
        ))
    return records


def search_uniprot_targets(question: str, *, limit: int = 20) -> list[TargetCandidate]:
    search_terms = _safe_search_terms(question)
    if not search_terms:
        return []
    queries = [f"organism_id:9606 AND reviewed:true AND ({search_terms})"]
    if " OR " not in search_terms:
        queries.append(f"organism_id:9606 AND reviewed:true AND keyword:\"{search_terms}\"")
    result_items: list[dict[str, Any]] = []
    seen_accessions: set[str] = set()
    for query in queries:
        payload = _request_json(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": query, "format": "json", "size": min(limit, 100), "fields": "accession,id,protein_name,gene_names,organism_name,reviewed"},
        )
        for item in payload.get("results", []):
            accession = item.get("primaryAccession") if isinstance(item, dict) else None
            if accession and accession not in seen_accessions:
                seen_accessions.add(accession)
                result_items.append(item)
            if len(result_items) >= limit:
                break
        if len(result_items) >= limit:
            break
    candidates: list[TargetCandidate] = []
    for item in result_items:
        if not isinstance(item, dict) or not item.get("primaryAccession"):
            continue
        protein_name = ((item.get("proteinDescription") or {}).get("recommendedName") or {}).get("fullName", {}).get("value")
        genes = [gene.get("geneName", {}).get("value") for gene in item.get("genes", []) if isinstance(gene, dict)]
        candidates.append(TargetCandidate(
            canonical_name=_bounded_text(protein_name or item.get("uniProtkbId") or item["primaryAccession"], 300),
            uniprot_accession=item["primaryAccession"], uniprot_reviewed=True,
            gene_symbols=[gene for gene in genes if gene], protein_id=item.get("uniProtkbId"),
        ))
    return candidates


def search_uniprot_targets_from_evidence(sources: Iterable[SourceRecord], *, limit: int = 20) -> list[TargetCandidate]:
    """Resolve explicit gene-like mentions found in bounded evidence text."""
    ignored = {"AND", "THE", "FOR", "WITH", "FROM", "THIS", "THAT", "HUMAN", "DISEASE", "PATIENT", "ASTHMA", "CANCER", "PARKINSON", "ALZHEIMER"}
    tokens: list[str] = []
    alias_patterns = {
        r"\binterleukin[- ]?5\b": "IL5",
        r"\binterleukin[- ]?5\s+receptor(?:\s+subunit)?\s+alpha\b": "IL5RA",
        r"\binterleukin[- ]?4\s+receptor(?:\s+subunit)?\s+alpha\b": "IL4R",
        r"\bthymic stromal lymphopoietin\b": "TSLP",
        r"\btumor necrosis factor\b": "TNF",
        r"\bglucagon[- ]like peptide[- ]1 receptor\b": "GLP1R",
        r"\bdipeptidyl[- ]peptidase[- ]4\b": "DPP4",
        r"\bperoxisome proliferator[- ]activated receptor gamma\b": "PPARG",
        r"\binterleukin[- ]5 receptor alpha\b": "IL5RA",
        r"\binterleukin[- ]4 receptor alpha\b": "IL4R",
        r"\bIL[- ]5R(?:alpha|α)\b": "IL5RA",
        r"\bIL[- ]4R(?:alpha|α)\b": "IL4R",
    }
    for source in sources:
        text = f"{source.title or ''} {source.abstract_or_excerpt or ''}"
        for pattern, alias in alias_patterns.items():
            if re.search(pattern, text, flags=re.IGNORECASE) and alias not in tokens:
                tokens.append(alias)
        for token in re.findall(r"\b[A-Z][A-Z0-9-]{1,9}\b", text):
            if token not in ignored and token not in tokens:
                tokens.append(token)
    candidates: list[TargetCandidate] = []
    seen: set[str] = set()
    for token in tokens[:40]:
        canonical_symbol = _hgnc_canonical_symbol(token) or token
        try:
            payload = _request_json("https://rest.uniprot.org/uniprotkb/search", params={"query": f"organism_id:9606 AND reviewed:true AND (gene_exact:{canonical_symbol} OR gene:{canonical_symbol})", "format": "json", "size": 2, "fields": "accession,id,protein_name,gene_names,organism_name,reviewed"})
        except TargetIntelligenceError:
            continue
        for item in payload.get("results", []):
            accession = item.get("primaryAccession") if isinstance(item, dict) else None
            if not accession or accession in seen: continue
            seen.add(accession)
            protein_name = ((item.get("proteinDescription") or {}).get("recommendedName") or {}).get("fullName", {}).get("value")
            genes = [g.get("geneName", {}).get("value") for g in item.get("genes", []) if isinstance(g, dict)]
            supporting = [source.stable_id for source in sources if token in f"{source.title or ''} {source.abstract_or_excerpt or ''}"]
            candidates.append(TargetCandidate(canonical_name=_bounded_text(protein_name or accession, 300), uniprot_accession=accession, uniprot_reviewed=True, gene_symbols=[g for g in genes if g], supporting_source_ids=supporting, literature_hits=len(supporting)))
            if len(candidates) >= limit: return candidates
    return candidates


@lru_cache(maxsize=512)
def _cached_chembl_enrichment(accession: str) -> dict[str, Any]:
    """Resolve ChEMBL target/activity data with a fallback query path."""
    queries = [
        {"target_components__accession": accession, "target_type": "SINGLE PROTEIN", "organism": "Homo sapiens", "limit": 10},
        {"target_components__accession": accession, "limit": 25},
    ]
    last_error: Exception | None = None
    targets: list[dict[str, Any]] = []
    for params in queries:
        try:
            payload = _request_json("https://www.ebi.ac.uk/chembl/api/data/target.json", params=params)
            targets = [item for item in payload.get("targets", []) if isinstance(item, dict)]
            if targets:
                break
        except TargetIntelligenceError as exc:
            last_error = exc
    if not targets:
        if last_error:
            raise TargetIntelligenceError("ChEMBL target lookup unavailable") from last_error
        return {"status": "not_found"}
    target = next((item for item in targets if item.get("organism") == "Homo sapiens" and item.get("target_type") == "SINGLE PROTEIN"), targets[0])
    target_id = target.get("target_chembl_id")
    if not target_id:
        return {"status": "not_found"}
    activity_count = None
    try:
        activity_payload = _request_json(
            "https://www.ebi.ac.uk/chembl/api/data/activity.json",
            params={"target_chembl_id": target_id, "limit": 1},
        )
        activity_count = int((activity_payload.get("page_meta") or {}).get("total_count") or 0)
    except TargetIntelligenceError:
        pass
    return {"status": "ok", "target_id": target_id, "target_name": target.get("pref_name"), "activity_count": activity_count}


def enrich_target_with_chembl(candidate: TargetCandidate) -> TargetCandidate:
    try:
        enrichment = _cached_chembl_enrichment(candidate.uniprot_accession)
    except TargetIntelligenceError:
        candidate.chembl_status = "temporarily_unavailable"
        candidate.chembl_activity_status = "not_checked"
        candidate.missing_data.append("ChEMBL target enrichment temporarily unavailable")
        return candidate
    if enrichment.get("status") != "ok":
        candidate.chembl_status = "not_found"
        candidate.chembl_activity_status = "not_found"
        candidate.missing_data.append("No human single-protein ChEMBL target mapping found")
        return candidate
    candidate.chembl_status = "ok"
    candidate.chembl_target_id = enrichment.get("target_id")
    candidate.chembl_target_name = enrichment.get("target_name")
    candidate.chembl_activity_count = enrichment.get("activity_count")
    if candidate.chembl_activity_count is None:
        candidate.chembl_activity_status = "temporarily_unavailable"
        candidate.missing_data.append("ChEMBL activity count temporarily unavailable")
    else:
        candidate.chembl_activity_status = "ok"
    return candidate


def _calculate_literature_hits(candidate: TargetCandidate, sources: Iterable[SourceRecord], question: str) -> None:
    aliases = [candidate.canonical_name.lower(), candidate.uniprot_accession.lower(), *[gene.lower() for gene in candidate.gene_symbols]]
    for source in sources:
        text = f"{source.title or ''} {source.abstract_or_excerpt or ''}".lower()
        alias_hit = any(alias and alias in text for alias in aliases)
        # Disease-term relevance is not direct target evidence. It may explain
        # why UniProt returned a candidate, but only a target alias in a source
        # record contributes to the direct literature-hit count.
        if alias_hit:
            candidate.literature_hits += 1
            candidate.supporting_source_ids.append(source.stable_id)
            if re.search(r"\b(biomarker|stratif|mutation|variant|expression|pathway|association)\b", text):
                if candidate.evidence_role == "unclassified":
                    candidate.evidence_role = "indirect_or_biomarker"
            elif re.search(r"\b(target|inhibitor|inhibition|therapy|therapeutic|antibody|drug)\b", text):
                candidate.evidence_role = "direct_or_modality_target"


def _modality_guard(candidate: TargetCandidate, sources: Iterable[SourceRecord]) -> tuple[str, list[str]]:
    """Classify modality conservatively before allowing QSAR handoff."""
    supporting = set(candidate.supporting_source_ids)
    text = " ".join(f"{source.title or ''} {source.abstract_or_excerpt or ''}" for source in sources if source.stable_id in supporting).lower()
    biologic_accessions = {"P01375", "P08887", "P13612", "P31358", "P11836", "P05113", "Q01344", "P24394", "Q969D9", "P19438"}
    if candidate.uniprot_accession in biologic_accessions or re.search(r"\b(monoclonal antibody|anti[- ](?:il|tnf)|biologic therapy|antibody against)\b", text):
        return "biologic_or_antibody_only", ["Biologic/antibody evidence requires a separate modality track"]
    if candidate.chembl_status != "ok":
        return "modality_unknown", ["ChEMBL single-protein mapping is unavailable"]
    if not candidate.chembl_activity_count or candidate.chembl_activity_count <= 0:
        return "modality_unknown", ["No ChEMBL activity records available"]
    return "small_molecule_candidate", []


def _score_candidate(candidate: TargetCandidate, weights: Mapping[str, float]) -> dict[str, Any]:
    direct_hits = min(candidate.literature_hits, 6)
    biological = min(100.0, 30.0 + direct_hits * 10.0 + (15.0 if candidate.uniprot_reviewed else 0.0))
    evidence = min(100.0, 20.0 + direct_hits * 8.0 + (20.0 if candidate.supporting_source_ids else 0.0))
    activity_count = candidate.chembl_activity_count
    qsar_unavailable = candidate.chembl_status == "temporarily_unavailable" or candidate.chembl_activity_status == "temporarily_unavailable"
    if qsar_unavailable:
        qsar = None
        candidate.missing_data.append("QSAR readiness unavailable because ChEMBL could not be checked")
    elif activity_count is None:
        qsar = 10.0
        candidate.missing_data.append("No ChEMBL activity count available")
    elif activity_count < 50:
        qsar = min(35.0, 10.0 + activity_count * 0.5)
        candidate.missing_data.append("Fewer than 50 ChEMBL activity records")
    else:
        qsar = min(100.0, 35.0 + math.log10(activity_count + 1) * 20.0)
    if qsar is None:
        available_weight = weights["biological_relevance"] + weights["evidence_quality"]
        composite = (biological * weights["biological_relevance"] + evidence * weights["evidence_quality"]) / available_weight
    else:
        composite = biological * weights["biological_relevance"] + evidence * weights["evidence_quality"] + qsar * weights["qsar_readiness"]
    labels = []
    if qsar is None:
        labels.append("QSAR readiness unavailable")
    elif qsar < 40.0:
        labels.append("Biologically relevant, QSAR-unready")
    if not candidate.supporting_source_ids:
        labels.append("Insufficient direct literature evidence")
    confidence = "high" if biological >= 70 and evidence >= 70 else "medium" if biological >= 45 else "low"
    return {
        "biological_relevance": round(biological, 2), "evidence_quality": round(evidence, 2),
        "qsar_readiness": round(qsar, 2) if qsar is not None else None, "composite": round(composite, 2), "confidence": confidence,
        "labels": labels,
        "explanations": {
            "literature_hits": candidate.literature_hits,
            "evidence_role": candidate.evidence_role,
            "chembl_activity_count": activity_count,
            "reviewed_human_uniprot": candidate.uniprot_reviewed,
            "chembl_status": candidate.chembl_status,
            "chembl_activity_status": candidate.chembl_activity_status,
        },
    }


def run_target_intelligence_search(question: str, *, max_targets: int = 10,
                                    ranking_weights: Mapping[str, Any] | None = None,
                                    source_types: Iterable[str] = DEFAULT_SOURCE_TYPES,
                                    source_limit: int = 10,
                                    llm_enhance: bool = False,
                                    additional_sources: Iterable[SourceRecord] = ()) -> dict[str, Any]:
    """Run deterministic retrieval/ranking against live public sources."""
    question = _bounded_text(question, 10000)
    if len(question) < 3:
        raise ValueError("question must contain at least 3 characters")
    if not 1 <= max_targets <= 10:
        raise ValueError("max_targets must be between 1 and 10")
    weights = normalize_weights(ranking_weights)
    requested = [source for source in source_types if source in DEFAULT_SOURCE_TYPES]
    sources: list[SourceRecord] = []
    source_errors: list[dict[str, str]] = []
    source_functions = {"europe_pmc": search_europe_pmc, "pubmed": search_pubmed, "openalex": search_openalex, "crossref": search_crossref}
    for source_type in requested:
        function = source_functions.get(source_type)
        if not function:
            continue
        try:
            sources.extend(function(question, limit=source_limit))
        except TargetIntelligenceError as exc:
            logger.warning("Target Intelligence source failed", extra={"source_type": source_type, "error": str(exc)})
            source_errors.append({"source_type": source_type, "error": str(exc)})
    sources.extend(list(additional_sources))
    try:
        candidates = search_uniprot_targets(question, limit=max_targets)
    except TargetIntelligenceError as exc:
        logger.warning("UniProt target retrieval failed", extra={"error": str(exc)})
        candidates = []
        source_errors.append({"source_type": "uniprot", "error": str(exc)})
    evidence_candidates = search_uniprot_targets_from_evidence(sources, limit=max_targets)
    existing = {candidate.uniprot_accession for candidate in candidates}
    candidates.extend(candidate for candidate in evidence_candidates if candidate.uniprot_accession not in existing)
    for candidate in candidates:
        _calculate_literature_hits(candidate, sources, question)

    # ChEMBL enrichment is network-bound. Keep it bounded and parallel so one
    # slow target cannot make the browser appear frozen for several minutes.
    def enrich(candidate: TargetCandidate) -> tuple[TargetCandidate, str | None]:
        try:
            return enrich_target_with_chembl(candidate), None
        except TargetIntelligenceError as exc:
            candidate.missing_data.append("ChEMBL target enrichment failed")
            return candidate, str(exc)

    enriched: list[TargetCandidate] = []
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(candidates)))) as executor:
        futures = [executor.submit(enrich, candidate) for candidate in candidates]
        for future in as_completed(futures):
            candidate, error = future.result()
            enriched.append(candidate)
            if error:
                source_errors.append({"source_type": "chembl", "error": error})

    ranked: list[dict[str, Any]] = [
        {"candidate": candidate, "scores": _score_candidate(candidate, weights)}
        for candidate in enriched
    ]
    ranked.sort(key=lambda item: item["scores"]["composite"], reverse=True)
    targets = []
    for rank, item in enumerate(ranked[:max_targets], start=1):
        candidate: TargetCandidate = item["candidate"]
        evidence_role = item["scores"].get("explanations", {}).get("evidence_role", candidate.evidence_role)
        modality, modality_reasons = _modality_guard(candidate, sources)
        if evidence_role == "direct_or_modality_target" and candidate.uniprot_reviewed and candidate.supporting_source_ids and modality == "small_molecule_candidate":
            normalization_status = "canonical_candidate"
        elif evidence_role in {"indirect_or_biomarker", "unclassified"}:
            normalization_status = "contextual_or_ambiguous"
        else:
            normalization_status = "review_required"
        targets.append({
            "rank": rank, "canonical_name": candidate.canonical_name,
            "identifiers": {"uniprot": candidate.uniprot_accession, "chembl_target": candidate.chembl_target_id},
            "gene_symbols": candidate.gene_symbols, "scores": item["scores"],
            "normalization": {"status": normalization_status, "confidence": item["scores"].get("confidence"), "evidence_role": evidence_role, "modality": modality, "guard_reasons": modality_reasons},
            "evidence": [source.stable_id for source in sources if source.stable_id in candidate.supporting_source_ids],
            "missing_data": sorted(set(candidate.missing_data)),
            "limitations": ["Deterministic retrieval/ranking only; LLM evidence extraction is separately labelled when enabled."],
        })
    report = {
        "report_version": "ti-retrieval-v1",
        "query": {"question": question, "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "provenance": {"sources": [asdict(source) for source in sources], "source_types_requested": requested, "source_errors": source_errors, "ranking": {"weights": weights, "score_version": "ti-score-v2"}, "software": {"component": "QSARify Target Intelligence", "retrieval_mode": "deterministic"}},
        "targets": targets,
        "limitations": ["Literature coverage depends on source indexing and available abstracts/full text.", "Target ranking is a research prioritisation heuristic, not a clinical or regulatory conclusion.", "QSAR readiness is an initial data-availability screen, not model validation."],
        "safety_notice": "Research support only; not medical advice or an OECD validation claim.",
    }
    if llm_enhance:
        try:
            from services.evidence_harness import extract_evidence

            harness = extract_evidence(question, sources)
            report["evidence_extraction"] = harness["extraction"]
            report["provenance"]["llm_harness"] = harness["metadata"]
        except Exception as exc:
            logger.warning("Evidence harness failed", extra={"error": str(exc)})
            report["provenance"]["llm_harness"] = {"status": "failed", "error": "LLM evidence extraction unavailable"}
            report["limitations"].append("LLM evidence extraction was unavailable for this run.")
    return report
