"""Extract an independent BindingDB target-matched IC50 validation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from services import curation as curation_service  # noqa: E402


TARGETS = {"AChE": "P22303", "BACE1": "P56817", "COX-2": "P35354", "MAO-B": "P27338", "VISFATIN": "P43490"}
NUMERIC_IC50 = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_training_smiles(path: Path) -> set[str]:
    frame = pd.read_csv(path, usecols=["smiles"], low_memory=False)
    return {value for value in frame["smiles"].map(curation_service.curate_smiles) if value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--training-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-nm", type=float, default=10000.0)
    args = parser.parse_args()

    input_path = args.input.resolve()
    training_path = args.training_input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_smiles = read_training_smiles(training_path)
    target_by_uniprot = {value: key for key, value in TARGETS.items()}
    accepted: list[dict] = []
    counters = {target: {"target_rows_seen": 0, "literature_rows": 0, "exact_ic50_rows": 0, "valid_curated_rows": 0, "training_overlap_rows": 0} for target in TARGETS}

    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        indices = {name: index for index, name in enumerate(header)}
        primary_columns = [name for name in header if "UniProt (SwissProt) Primary ID of Target Chain" in name]
        required = ["Ligand SMILES", "IC50 (nM)", "Target Name", "Target Source Organism According to Curator or DataSource", "Curation/DataSource", "Date of publication", "Date in BindingDB"]
        missing = [name for name in required if name not in indices]
        if missing or not primary_columns:
            raise ValueError(f"BindingDB header is missing required fields: {missing}")

        for row in reader:
            if len(row) < len(header):
                continue
            target = next((target_by_uniprot.get(row[indices[column]].strip()) for column in primary_columns if row[indices[column]].strip() in target_by_uniprot), None)
            if not target:
                continue
            counters[target]["target_rows_seen"] += 1
            source = row[indices["Curation/DataSource"]].strip()
            source_lower = source.lower()
            if "curated from the literature" not in source_lower and "us patent" not in source_lower:
                continue
            counters[target]["literature_rows"] += 1
            match = NUMERIC_IC50.match(row[indices["IC50 (nM)"]].strip())
            if not match:
                continue
            counters[target]["exact_ic50_rows"] += 1
            curated = curation_service.curate_smiles(row[indices["Ligand SMILES"]].strip())
            if not curated:
                continue
            counters[target]["valid_curated_rows"] += 1
            if curated in training_smiles:
                counters[target]["training_overlap_rows"] += 1
                continue
            value = float(match.group(1))
            def get(name: str) -> str:
                return row[indices[name]].strip() if name in indices else ""
            accepted.append({
                "target_name": target,
                "uniprot_id": TARGETS[target],
                "smiles": curated,
                "standard_type": "IC50",
                "standard_value": value,
                "standard_units": "nM",
                "standard_relation": "=",
                "bioactivity_class": "Active" if value <= args.threshold_nm else "Inactive",
                "target_name_bindingdb": get("Target Name"),
                "target_organism_bindingdb": get("Target Source Organism According to Curator or DataSource"),
                "curation_source": source,
                "publication_date": get("Date of publication"),
                "bindingdb_date": get("Date in BindingDB"),
                "article_doi": get("Article DOI"),
                "bindingdb_entry_doi": get("BindingDB Entry DOI"),
                "pmid": get("PMID"),
                "bindingdb_reactant_set_id": get("BindingDB Reactant_set_id"),
                "bindingdb_monomer_id": get("BindingDB MonomerID"),
            })

    result = pd.DataFrame(accepted).drop_duplicates(subset=["target_name", "smiles", "standard_value", "pmid", "article_doi"])
    if result.empty:
        raise RuntimeError("No independent BindingDB rows survived the filters.")
    result.to_csv(output_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_EXTERNAL_VALIDATION_SET",
        "source": "BindingDB",
        "source_release": "BindingDB_All_202608",
        "source_policy": "BindingDB-curated literature and US-patent rows; ChEMBL/PubChem-derived rows excluded",
        "input": {"path": input_path.as_posix(), "sha256": sha256_file(input_path)},
        "training_input": {"path": training_path.as_posix(), "sha256": sha256_file(training_path)},
        "output": {"path": output_path.as_posix(), "sha256": sha256_file(output_path), "rows": int(len(result))},
        "threshold_nm": args.threshold_nm,
        "targets": counters,
        "target_rows_output": {str(key): int(value) for key, value in result["target_name"].value_counts().sort_index().items()},
        "claim_boundary": "Candidate independent validation set; endpoint/assay comparability and expert review remain required.",
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
