"""Bounded, provenance-preserving user reference inputs for Target Intelligence."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
from typing import Any, Iterable

from services.target_intelligence import SourceRecord, _request_json


MAX_REFERENCE_INPUTS = 10
MAX_REFERENCE_LENGTH = 25_000
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[^\s]+$", re.IGNORECASE)
PMID_PATTERN = re.compile(r"^(?:PMID\s*:\s*)?(\d+)$", re.IGNORECASE)


class ReferenceInputError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(source_type: str, stable_id: str, *, title: str | None = None, url: str | None = None,
            doi: str | None = None, year: int | None = None, tier: str = "E",
            excerpt: str | None = None, metadata: dict[str, Any] | None = None) -> SourceRecord:
    return SourceRecord(
        source_type=source_type,
        stable_id=stable_id,
        title=title,
        url=url,
        doi=doi,
        publication_year=year,
        evidence_tier=tier,
        abstract_or_excerpt=excerpt,
        retrieved_at=_now(),
        source_snapshot_hash=_source_hash(repr(sorted((metadata or {}).items()))),
        metadata=metadata or {},
    )


def _normalize_input(value: Any) -> str:
    normalized = " ".join(str(value or "").split()).strip()
    if not normalized:
        raise ReferenceInputError("Reference input cannot be empty")
    if len(normalized) > MAX_REFERENCE_LENGTH:
        raise ReferenceInputError(f"Reference input exceeds {MAX_REFERENCE_LENGTH} characters")
    return normalized


def _doi_from_value(value: str) -> str | None:
    candidate = value.strip()
    if candidate.lower().startswith("doi:"):
        candidate = candidate[4:].strip()
    parsed = urlparse(candidate)
    if parsed.scheme.lower() == "https" and parsed.netloc.lower() in {"doi.org", "dx.doi.org"}:
        candidate = parsed.path.lstrip("/")
    return candidate if DOI_PATTERN.fullmatch(candidate) else None


def _pmid_from_value(value: str) -> str | None:
    candidate = value.strip()
    parsed = urlparse(candidate)
    if parsed.scheme.lower() == "https" and parsed.netloc.lower() in {"pubmed.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov"}:
        match = re.search(r"/(\d+)(?:/|$)", parsed.path)
        if match:
            return match.group(1)
    match = PMID_PATTERN.fullmatch(candidate)
    return match.group(1) if match else None


def _validate_public_https_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ReferenceInputError("Publisher references must use an HTTPS URL")
    host = parsed.hostname.strip("[]").lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Hostnames are recorded only; no user-controlled URL is fetched here.
        if host in {"localhost", "metadata.google.internal"} or host.endswith(".local"):
            raise ReferenceInputError("Private or local publisher URLs are not allowed")
    else:
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            raise ReferenceInputError("Private or local publisher URLs are not allowed")
    return value


def resolve_reference_input(value: Any) -> SourceRecord:
    normalized = _normalize_input(value)
    doi = _doi_from_value(normalized)
    if doi:
        payload = _request_json(f"https://api.crossref.org/works/{quote(doi, safe='')}", timeout=15, retries=1)
        item = payload.get("message") or {}
        title = (item.get("title") or [None])[0]
        date_parts = ((item.get("published") or {}).get("date-parts") or [[None]])[0]
        year = date_parts[0] if date_parts and isinstance(date_parts[0], int) else None
        return _source("crossref", f"DOI:{doi}", title=title or doi, url=item.get("URL") or f"https://doi.org/{doi}", doi=doi,
                       year=year, tier="C", metadata={"user_reference": True, "input_kind": "doi"})

    pmid = _pmid_from_value(normalized)
    if pmid:
        payload = _request_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "json"},
            timeout=15,
            retries=1,
        )
        item = (payload.get("result") or {}).get(pmid) or {}
        return _source(
            "pubmed",
            f"PMID:{pmid}",
            title=item.get("title") or f"PubMed record {pmid}",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            year=None,
            tier="E",
            metadata={"user_reference": True, "input_kind": "pmid", "journal": item.get("fulljournalname")},
        )

    parsed = urlparse(normalized)
    if parsed.scheme.lower() in {"http", "https"}:
        url = _validate_public_https_url(normalized)
        return _source(
            "user_text",
            f"URL:{_source_hash(url)[:24]}",
            title="User-provided publisher URL",
            url=url,
            tier="E",
            metadata={"user_reference": True, "input_kind": "publisher_url", "content_fetched": False},
        )

    return _source(
        "user_text",
        f"USER_TEXT:{_source_hash(normalized)[:24]}",
        title="User-provided pasted text",
        excerpt=normalized,
        tier="E",
        metadata={"user_reference": True, "input_kind": "pasted_text"},
    )


def resolve_reference_inputs(values: Iterable[Any]) -> tuple[list[SourceRecord], list[dict[str, str]]]:
    values = list(values)
    if len(values) > MAX_REFERENCE_INPUTS:
        raise ReferenceInputError(f"At most {MAX_REFERENCE_INPUTS} reference inputs are allowed")
    sources: list[SourceRecord] = []
    errors: list[dict[str, str]] = []
    for index, value in enumerate(values):
        try:
            sources.append(resolve_reference_input(value))
        except Exception as exc:
            errors.append({"source_type": "user_reference", "error": f"Reference {index + 1}: {str(exc)[:300]}"})
    return sources, errors
