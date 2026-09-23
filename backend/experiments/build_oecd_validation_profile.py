"""Build an OECD/QMRF-oriented readiness profile for each target endpoint.

This command deliberately fails to claim validation when required endpoint,
assay, external-predictivity, or mechanistic evidence is absent. It produces a
machine-readable checklist and one QMRF-style profile per target so that an
independent validation dataset can be added without changing the reporting
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TARGETS = ["AChE", "BACE1", "COX-2", "MAO-B", "VISFATIN"]
REQUIRED_PROVENANCE_COLUMNS = {
    "standard_type",
    "standard_value",
    "standard_units",
    "standard_relation",
    "assay_chembl_id",
    "assay_type",
    "assay_description",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_coverage(frame: pd.DataFrame, column: str) -> dict:
    if column not in frame.columns:
        return {"present": False, "non_missing": 0, "coverage": 0.0, "unique_values": []}
    values = frame[column]
    non_missing = values.notna() & values.astype("string").str.strip().ne("")
    unique_values = sorted(values.loc[non_missing].astype(str).unique().tolist())
    return {
        "present": True,
        "non_missing": int(non_missing.sum()),
        "coverage": float(non_missing.mean()) if len(values) else 0.0,
        "unique_values": unique_values[:100],
    }


def qmr_profile(target: str, target_frame: pd.DataFrame, *, input_path: Path, external_available: bool) -> dict:
    missing = sorted(REQUIRED_PROVENANCE_COLUMNS - set(target_frame.columns))
    standard_units = value_coverage(target_frame, "standard_units")
    standard_type = value_coverage(target_frame, "standard_type")
    relation = value_coverage(target_frame, "standard_relation")
    assay_id = value_coverage(target_frame, "assay_chembl_id")
    assay_type = value_coverage(target_frame, "assay_type")
    assay_description = value_coverage(target_frame, "assay_description")

    principles = {
        "defined_endpoint": {
            "status": "INCOMPLETE" if missing or not standard_type["present"] else "READY_FOR_AUDIT",
            "evidence": {
                "target": target,
                "activity_type_expected": "IC50",
                "units_expected": "nM",
                "threshold_nm": 10000.0,
                "provenance_columns_missing": missing,
                "standard_type": standard_type,
                "standard_units": standard_units,
                "standard_relation": relation,
            },
        },
        "unambiguous_algorithm": {
            "status": "READY",
            "evidence": "Versioned QSARify feature, preprocessing, estimator, seed, and model-artifact contracts are recorded separately.",
        },
        "defined_domain_of_applicability": {
            "status": "PARTIAL",
            "evidence": "Target-specific Morgan/Tanimoto AD references and status-stratified diagnostics exist; threshold calibration and prospective coverage/error validation remain incomplete.",
        },
        "goodness_robustness_predictivity": {
            "status": "PARTIAL" if not external_available else "READY_FOR_EXTERNAL_AUDIT",
            "evidence": "Internal scaffold-disjoint repeated evaluation, MCC, balanced accuracy, y-randomization, baselines, and bootstrap diagnostics are archived; an independent temporal/external test set is required.",
        },
        "mechanistic_interpretation": {
            "status": "PRELIMINARY",
            "evidence": "Target context and descriptive feature importances are documented, but no causal or endpoint-specific mechanistic interpretation has been validated for this profile.",
        },
    }

    overall = "NOT_VALIDATED"
    if not missing and external_available and all(item["status"] == "READY" for item in principles.values()):
        overall = "READY_FOR_EXPERT_REVIEW"
    return {
        "target": target,
        "overall_status": overall,
        "model_scope": "target-specific binary IC50 activity model",
        "input": {"path": input_path.as_posix(), "sha256": sha256_file(input_path), "rows": int(len(target_frame))},
        "assay_provenance_coverage": {
            "assay_chembl_id": assay_id,
            "assay_type": assay_type,
            "assay_description": assay_description,
        },
        "principles": principles,
        "required_next_evidence": [
            "Retain endpoint and assay provenance fields in the archived input.",
            "Freeze the target-specific model and preprocessing before external testing.",
            "Provide an independent temporal or external target-matched validation set.",
            "Validate AD coverage and error rates on the independent set.",
            "Add endpoint-specific mechanistic interpretation where scientifically defensible.",
            "Complete expert/QMRF review before any OECD-oriented claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-input", type=Path, default=None)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, low_memory=False)
    profiles = {
        target: qmr_profile(
            target,
            frame.loc[frame["target_name"] == target].copy(),
            input_path=input_path,
            external_available=args.external_input is not None and args.external_input.is_file(),
        )
        for target in TARGETS
    }
    matrix = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": "OECD QSAR validation principles / QMRF-oriented readiness profile",
        "claim_boundary": "This output is a readiness assessment, not an OECD validation or regulatory approval.",
        "input": {"path": input_path.as_posix(), "sha256": sha256_file(input_path), "rows": int(len(frame))},
        "external_input": str(args.external_input.resolve()) if args.external_input and args.external_input.is_file() else None,
        "profiles": profiles,
        "overall_status": "NOT_VALIDATED",
    }
    (output_dir / "validation_matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for target, profile in profiles.items():
        rows.append({"target": target, "overall_status": profile["overall_status"], **{key: value["status"] for key, value in profile["principles"].items()}})
        qmrf = [
            f"# QMRF-oriented profile: {target}",
            "",
            "Status: NOT OECD-VALIDATED",
            "",
            "This profile documents readiness against the OECD QSAR principles. It is not a regulatory validation or approval.",
            "",
            "## Endpoint and provenance",
            "",
            json.dumps(profile["principles"]["defined_endpoint"], indent=2),
            "",
            "## OECD principle checklist",
            "",
        ]
        for principle, item in profile["principles"].items():
            qmrf.append(f"- **{principle}**: {item['status']} — {item['evidence']}")
        qmrf += ["", "## Required evidence before an OECD-oriented claim", ""]
        qmrf.extend(f"- {item}" for item in profile["required_next_evidence"])
        (output_dir / f"QMRF_{target.replace('-', '_')}.md").write_text("\n".join(qmrf) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "validation_matrix.csv", index=False)
    (output_dir / "README.md").write_text(
        "# OECD-oriented validation profiles\n\n"
        "These target-specific profiles are deliberately fail-closed. They document what QSARify can evidence and what remains required; they do not claim OECD validation.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_dir), "targets": TARGETS, "status": "NOT_VALIDATED"}, indent=2))


if __name__ == "__main__":
    main()
