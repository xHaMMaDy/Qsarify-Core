"""Held-out hierarchical evidence/modality reranking experiment."""
from __future__ import annotations
import csv,json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

ROOT=Path(__file__).resolve().parents[2]
POS={"direct_target","disease_associated_protein","biologic_or_antibody_only"}
def main():
 rows=list(csv.DictReader((ROOT/'expert-review/target-intelligence-independent-review-2026-09-16/candidate_role_review_set.v3.expanded-800.reviewed.csv').open(encoding='utf-8-sig')))
 for r in rows:
  r['label']=int(r['reviewed_role'] in POS)
  for k in ('rank','source_evidence_count','biological_relevance','evidence_quality','qsar_readiness'):
   try:r[k]=float(r[k])
   except:r[k]=0.0
 numeric=['rank','source_evidence_count','biological_relevance','evidence_quality','qsar_readiness']; cat=['evidence_role_suggested']
 pipe=Pipeline([('prep',ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])),('model',LogisticRegression(max_iter=1000,class_weight='balanced'))])
 cases=sorted({r['case_id'] for r in rows}); probs={}
 for held in cases:
  train=[r for r in rows if r['case_id']!=held]; test=[r for r in rows if r['case_id']==held]
  pipe.fit(pd.DataFrame(train),[r['label'] for r in train]); p=pipe.predict_proba(pd.DataFrame(test))[:,list(pipe.classes_).index(1)] if 1 in pipe.classes_ else [0.0]*len(test)
  for r,v in zip(test,p):probs[(r['case_id'],r['uniprot_accession'])]=float(v)
 reports=[]
 for line in (ROOT/'docs/target-intelligence/benchmark_reports.v1-frozen-source.jsonl').open(encoding='utf-8'):
  x=json.loads(line); x['report']['targets'].sort(key=lambda t:probs.get((x['case_id'],(t.get('identifiers') or {}).get('uniprot')),0),reverse=True)
  for i,t in enumerate(x['report']['targets'],1):t['rank']=i
  reports.append(x)
 out=ROOT/'docs/target-intelligence/benchmark_reports.v1-frozen-hierarchical.jsonl'; out.write_text(''.join(json.dumps(x)+'\n' for x in reports),encoding='utf-8'); print('wrote',out)
if __name__=='__main__':main()
