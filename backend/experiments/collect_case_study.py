"""Collect the QSARify five-target case-study data from ChEMBL.

The default mode writes a manifest only. Use ``--fetch`` to retrieve the full
activity and molecule records through the same functions used by the Flask
Data Collection service. Every output records the retrieval timestamp, target
metadata, filtering parameters, row counts, and SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app  # noqa: E402


KNOWN_TARGETS = {
    "MAO-B": {"uniprot_id": "P27338", "chembl_target_id": "CHEMBL2039", "pref_name": "Amine oxidase [flavin-containing] B"},
    "COX-2": {"uniprot_id": "P35354", "chembl_target_id": "CHEMBL230", "pref_name": "Prostaglandin G/H synthase 2"},
    "VISFATIN": {"uniprot_id": "P43490", "chembl_target_id": "CHEMBL1744525", "pref_name": "Nicotinamide phosphoribosyltransferase"},
    "BACE1": {"uniprot_id": "P56817", "chembl_target_id": "CHEMBL4822", "pref_name": "Beta-secretase 1"},
    "AChE": {"uniprot_id": "P22303", "chembl_target_id": "CHEMBL220", "pref_name": "Acetylcholinesterase"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_progress(generator):
    result = None
    for item in generator:
        if isinstance(item, str):
            continue
        result = item
    return result or []


def target_activity_count(target_id: str, activity_type: str) -> int:
    response = requests.get(
        f"{app.CHEMBL_API_URL}/activity.json",
        params={
            "target_chembl_id": target_id,
            "standard_type": activity_type,
            "standard_units": "nM",
            "limit": 1,
        },
        timeout=30,
    )
    response.raise_for_status()
    return int(response.json().get("page_meta", {}).get("total_count", 0))


def build_manifest(activity_type: str, threshold_nm: float) -> dict:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "database": "ChEMBL",
            "api_base": app.CHEMBL_API_URL,
            "activity_type": activity_type,
            "standard_units": "nM",
            "threshold_nm": threshold_nm,
            "retrieval_policy": "target accession resolution followed by activity pagination",
            "query_parameters": {
                "standard_type": activity_type,
                "standard_units": "nM",
                "page_limit": 1000,
            "molecule_batch_limit": 100,
            "retained_activity_fields": [
                "activity_id",
                "molecule_chembl_id",
                "standard_type",
                "standard_value",
                "standard_units",
                "standard_relation",
                "assay_chembl_id",
                "assay_type",
                "assay_description",
                "document_chembl_id",
                "src_id",
                "data_validity_comment",
            ],
            },
            "assay_qualifier_policy": "ChEMBL activity qualifiers and assay-context metadata are retained in source rows when available but are not harmonised for the pooled benchmark.",
            "database_version_policy": "ChEMBL API release is not pinned; retrieval timestamp and per-file SHA-256 checksums define the archived snapshot.",
        },
        "analysis_policy": {
            "activity_units_required": "nM",
            "positive_standard_value_required": True,
            "smiles_policy": "RDKit parse and canonicalize; invalid structures excluded",
            "duplicate_key": "target_name + canonical_smiles",
            "duplicate_policy": "median standard_value per target/compound, then reapply the 10000 nM threshold",
            "conflicting_binary_labels": "resolved by the median standard_value rather than choosing a row",
        },
        "targets": [],
    }
    for name, uniprot_id in app.PROTEIN_MAP.items():
        try:
            target = app.get_target(uniprot_id)
        except requests.RequestException:
            fallback = KNOWN_TARGETS[name]
            target = {
                **fallback,
                "target_chembl_id": fallback["chembl_target_id"],
                "organism": "Homo sapiens",
                "target_type": "SINGLE PROTEIN",
            }
        chembl_id = target["target_chembl_id"]
        try:
            activity_count = target_activity_count(chembl_id, activity_type)
        except requests.RequestException:
            activity_count = None
        manifest["targets"].append(
            {
                "name": name,
                "uniprot_id": uniprot_id,
                "chembl_target_id": chembl_id,
                "pref_name": target.get("pref_name"),
                "organism": target.get("organism"),
                "target_type": target.get("target_type"),
                "activity_count_at_manifest_time": activity_count,
            }
        )
    return manifest


def fetch_target_rows(target: dict, activity_type: str, threshold_nm: float) -> pd.DataFrame:
    activities = collect_progress(app.get_all_bioactivities(target["chembl_target_id"], activity_type))
    molecule_ids = [activity.get("molecule_chembl_id") for activity in activities]
    molecules = fetch_molecules_in_batches(molecule_ids)
    rows = app.process_and_merge_data(activities, molecules, threshold_nm)
    frame = pd.DataFrame(rows)
    frame.insert(0, "target_name", target["name"])
    frame.insert(1, "uniprot_id", target["uniprot_id"])
    frame.insert(2, "chembl_target_id", target["chembl_target_id"])
    return frame


def fetch_molecule_batch(molecule_ids: list[str]) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                f"{app.CHEMBL_API_URL}/molecule.json",
                params={
                    "molecule_chembl_id__in": ",".join(molecule_ids),
                    "limit": len(molecule_ids),
                },
                timeout=30,
            )
            response.raise_for_status()
            return {
                molecule["molecule_chembl_id"]: molecule
                for molecule in response.json().get("molecules", [])
                if molecule.get("molecule_chembl_id")
            }
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"ChEMBL molecule batch failed after 3 attempts: {last_error}")


def fetch_molecules_in_batches(molecule_ids: list[str], batch_size: int = 100) -> dict:
    unique_ids = sorted({item for item in molecule_ids if item})
    batches = [unique_ids[start : start + batch_size] for start in range(0, len(unique_ids), batch_size)]
    molecule_map = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_molecule_batch, batch) for batch in batches]
        for future in as_completed(futures):
            molecule_map.update(future.result())
    return molecule_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--activity-type", default="IC50")
    parser.add_argument("--threshold-nm", type=float, default=10000.0)
    parser.add_argument("--fetch", action="store_true", help="Fetch full activity and molecule records")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.activity_type, args.threshold_nm)

    if args.fetch:
        frames = []
        for target in manifest["targets"]:
            frame = fetch_target_rows(target, args.activity_type, args.threshold_nm)
            target["rows_fetched"] = int(len(frame))
            target["valid_smiles_rows"] = int(frame["smiles"].notna().sum())
            target["active_rows"] = int((frame["bioactivity_class"] == "Active").sum())
            target["inactive_rows"] = int((frame["bioactivity_class"] == "Inactive").sum())
            target_path = output_dir / f"{target['name'].lower().replace('-', '_')}.csv"
            frame.to_csv(target_path, index=False)
            target["file"] = target_path.name
            target["sha256"] = sha256_file(target_path)
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined_path = output_dir / "case_study_all_targets.csv"
        combined.to_csv(combined_path, index=False)
        manifest["combined_file"] = combined_path.name
        manifest["combined_sha256"] = sha256_file(combined_path)
        manifest["combined_rows"] = int(len(combined))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "fetch": args.fetch}, indent=2))


if __name__ == "__main__":
    main()
