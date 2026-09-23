"""Snapshot reviewed PubMed evidence for the frozen benchmark."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import requests
import xml.etree.ElementTree as ET


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cases = list(csv.DictReader((root / "docs/target-intelligence/benchmark_cases.v1-frozen.csv").open(encoding="utf-8-sig")))
    out = root / "docs/target-intelligence/frozen_source_snapshots.v1.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for case in cases:
            ids = [x for x in (case.get("evidence_pmids") or "").split("|") if x]
            payload = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=60).json()
            abstract_xml = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}, timeout=60).text
            abstracts = {}
            root_xml = ET.fromstring(abstract_xml)
            for article in root_xml.findall(".//PubmedArticle"):
                pmid_node = article.find(".//PMID")
                if pmid_node is None: continue
                abstracts[pmid_node.text] = " ".join("".join(x.itertext()) for x in article.findall(".//AbstractText"))
            rows = []
            for pmid in ids:
                item = (payload.get("result") or {}).get(str(pmid)) or {}
                rows.append({"stable_id": f"PMID:{pmid}", "title": item.get("title"), "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "publication_year": item.get("pubdate"), "source_type": "pubmed", "abstract_or_excerpt": abstracts.get(str(pmid), "")})
            handle.write(json.dumps({"case_id": case["case_id"], "question": case["question"], "sources": rows}, ensure_ascii=False) + "\n")
    print(f"Wrote {out} with {len(cases)} case snapshots")


if __name__ == "__main__":
    main()
