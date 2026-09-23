"""Build an expanded candidate-role review set from frozen evidence mentions."""
from __future__ import annotations
import csv, json
from pathlib import Path
from services.target_intelligence import SourceRecord, search_uniprot_targets, search_uniprot_targets_from_evidence

def main():
    root=Path(__file__).resolve().parents[2]
    reviewed=list(csv.DictReader((root/'expert-review/target-intelligence-independent-review-2026-09-16/candidate_role_review_set.v1.reviewed.csv').open(encoding='utf-8-sig')))
    existing={(r['case_id'],r['uniprot_accession']) for r in reviewed}
    frozen=list(csv.DictReader((root/'docs/target-intelligence/benchmark_cases.v1-frozen.csv').open(encoding='utf-8-sig')))
    snapshots={json.loads(x)['case_id']:json.loads(x) for x in (root/'docs/target-intelligence/frozen_source_snapshots.v1.jsonl').open(encoding='utf-8')}
    rows=list(reviewed)
    for case in frozen:
        sources=[SourceRecord(source_type=s['source_type'],stable_id=s['stable_id'],title=s.get('title'),abstract_or_excerpt=s.get('abstract_or_excerpt')) for s in snapshots[case['case_id']]['sources']]
        candidates = search_uniprot_targets_from_evidence(sources,limit=100)
        candidates += search_uniprot_targets(case["question"], limit=100)
        for rank, candidate in enumerate(candidates, start=11):
            key=(case['case_id'],candidate.uniprot_accession)
            if key in existing: continue
            existing.add(key)
            rows.append({'benchmark_version':'ti-benchmark-v1-frozen','case_id':case['case_id'],'rank':rank,'canonical_name':candidate.canonical_name,'gene_symbols':'|'.join(candidate.gene_symbols),'uniprot_accession':candidate.uniprot_accession,'chembl_target':'','evidence_role_suggested':candidate.evidence_role,'source_evidence_count':candidate.literature_hits,'biological_relevance':'','evidence_quality':'','qsar_readiness':'','frozen_inclusion_match':'','reviewed_role':'','review_confidence':'','review_notes':''})
    out=root/'docs/target-intelligence/candidate_role_review_set.v3.expanded-800.csv'
    with out.open('w',encoding='utf-8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Wrote {out} rows={len(rows)} added={len(rows)-len(reviewed)}')
if __name__=='__main__': main()
