"""Apply pre-specified semantic assay filters to the BindingDB candidate set."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FILTERS = {
    "AChE": [
        ("radical_scavenging", r"dpph|abts|radical\s*scaveng|antioxid"),
        ("BuChE_mixed", r"buch[eE]|butyrylcholinesterase|acetylcholinesterase\s*(?:and|&)\s*buch[eE]"),
    ],
    "BACE1": [
        ("BACE2_mixed_or_isoform", r"bace\s*2|bace2|beta[-\s]?secretase\s*2"),
        ("construct_variant", r"\[[0-9]+[-–][0-9]+\]|construct|truncated"),
    ],
    "COX-2": [
        ("mutant_E165G_or_other_mutant", r"e165g|mutant|mutated"),
        ("COX1_or_ovine_mixed", r"cox[-\s]?1|cyclooxygenase[-\s]?1|ovine"),
    ],
    "MAO-B": [
        ("zebrafish_behavioral", r"zebrafish|photomotor|\bpmr\b"),
        ("binding_or_radioligand", r"radioligand|thk[-\s]?5351|binding assay|competitive binding"),
        ("MAO_A_or_pan_MAO", r"mao[-\s]?a|mao\s*a/b|both mao|pan[-\s]?mao|monoamine oxidase\s*a"),
    ],
    "VISFATIN": [
        ("HDAC_misjoined", r"\bhdac\b|histone deacetylase"),
        ("single_source_patent_family", r"us patent"),
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, low_memory=False)
    frame["_audit_row_id"] = range(len(frame))
    frame["_exclusion_reason"] = ""
    frame["_assay_text"] = (
        frame["assay_description"].fillna("").astype(str)
        + " "
        + frame["assay_name"].fillna("").astype(str)
        + " "
        + frame["target_name_bindingdb"].fillna("").astype(str)
        + " "
        + frame["curation_source"].fillna("").astype(str)
    ).str.lower()

    for target, rules in FILTERS.items():
        target_mask = frame["target_name"].eq(target)
        for reason, expression in rules:
            matched = target_mask & frame["_exclusion_reason"].eq("") & frame["_assay_text"].str.contains(expression, regex=True, na=False)
            frame.loc[matched, "_exclusion_reason"] = reason

    excluded = frame.loc[frame["_exclusion_reason"].ne("")].copy()
    kept = frame.loc[frame["_exclusion_reason"].eq("")].copy()
    excluded.to_csv(output_path.with_name(output_path.stem + "_excluded.csv"), index=False)
    kept.drop(columns=["_audit_row_id", "_exclusion_reason", "_assay_text"], errors="ignore").to_csv(output_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": input_path.as_posix(), "sha256": sha256_file(input_path), "rows": int(len(frame))},
        "output": {"path": output_path.as_posix(), "sha256": sha256_file(output_path), "rows": int(len(kept))},
        "excluded_output": {"path": output_path.with_name(output_path.stem + "_excluded.csv").as_posix(), "sha256": sha256_file(output_path.with_name(output_path.stem + "_excluded.csv")), "rows": int(len(excluded))},
        "filters": {target: [reason for reason, _ in rules] for target, rules in FILTERS.items()},
        "excluded_by_target_and_reason": {
            str(target): {str(reason): int(count) for reason, count in group["_exclusion_reason"].value_counts().items()}
            for target, group in excluded.groupby("target_name", sort=True)
        },
        "remaining_by_target": {str(target): int(count) for target, count in kept["target_name"].value_counts().sort_index().items()},
        "claim_boundary": "Semantic cleaning is rule-based and pre-specified; expert review remains required.",
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
