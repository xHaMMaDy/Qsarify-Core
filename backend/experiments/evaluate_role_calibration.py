"""Held-out case-split calibration using reviewer role labels."""
from __future__ import annotations
import csv, json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
POSITIVE = {"direct_target", "disease_associated_protein", "biologic_or_antibody_only"}

def main():
    reviewed_path = ROOT / "expert-review/target-intelligence-independent-review-2026-09-16/candidate_role_review_set.v3.expanded-800.reviewed.csv"
    if not reviewed_path.exists():
        reviewed_path = ROOT / "expert-review/target-intelligence-independent-review-2026-09-16/candidate_role_review_set.v1.reviewed.csv"
    rows = list(csv.DictReader(reviewed_path.open(encoding="utf-8-sig")))
    for row in rows:
        row["label"] = int(row["reviewed_role"] in POSITIVE)
        for key in ("rank", "source_evidence_count", "biological_relevance", "evidence_quality", "qsar_readiness"):
            try: row[key] = float(row[key])
            except (TypeError, ValueError): row[key] = 0.0
    numeric = ["rank", "source_evidence_count", "biological_relevance", "evidence_quality", "qsar_readiness"]
    categorical = ["evidence_role_suggested"]
    pipe = Pipeline([("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    cases = sorted({r["case_id"] for r in rows})
    all_true, all_pred, all_prob = [], [], []
    ranked = []
    for held_out in cases:
        train = [r for r in rows if r["case_id"] != held_out]
        test = [r for r in rows if r["case_id"] == held_out]
        train_frame = pd.DataFrame(train); test_frame = pd.DataFrame(test)
        pipe.fit(train_frame, [r["label"] for r in train])
        probs = pipe.predict_proba(test_frame)[:, list(pipe.classes_).index(1)] if 1 in pipe.classes_ else [0.0] * len(test)
        for row, prob in zip(test, probs):
            all_true.append(row["label"]); all_prob.append(float(prob)); all_pred.append(int(prob >= 0.5)); ranked.append({"case_id": held_out, "uniprot": row["uniprot_accession"], "probability": round(float(prob), 5), "reviewed_role": row["reviewed_role"]})
    result = {"cases": len(cases), "rows": len(rows), "held_out_role_f1": f1_score(all_true, all_pred), "held_out_role_roc_auc": roc_auc_score(all_true, all_prob), "positive_roles": sorted(POSITIVE), "ranking": ranked}
    out = ROOT / "docs/target-intelligence/role_calibration_heldout_expanded_2026-09-23.json"; out.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps({k: result[k] for k in ("cases", "rows", "held_out_role_f1", "held_out_role_roc_auc")}))

if __name__ == "__main__": main()
