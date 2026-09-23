"""Join BindingDB reaction-set and assay descriptions onto the candidate set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reaction-map", type=Path, required=True)
    parser.add_argument("--assays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reaction_map: dict[str, str] = {}
    with args.reaction_map.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            reaction_map[row["REACTANT_SET_ID"]] = row["ENTRYID_ASSAYID"]

    assays: dict[str, dict[str, str]] = {}
    with args.assays.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = f"{row['ENTRYID']}_{row['ASSAYID']}"
            assays[key] = {
                "assay_entry_id": row["ENTRYID"],
                "assay_id": row["ASSAYID"],
                "assay_name": row["ASSAY_NAME"],
                "assay_description": row["DESCRIPTION"],
            }

    frame = pd.read_csv(args.input, low_memory=False)
    frame["bindingdb_entry_assay_id"] = frame["bindingdb_reactant_set_id"].astype(str).map(reaction_map).fillna("")
    for column in ("assay_entry_id", "assay_id", "assay_name", "assay_description"):
        frame[column] = frame["bindingdb_entry_assay_id"].map(lambda key: assays.get(key, {}).get(column, ""))
    frame.to_csv(args.output, index=False)
    coverage = float(frame["assay_description"].astype("string").str.strip().ne("").mean())
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input": {"path": args.input.resolve().as_posix(), "sha256": sha256_file(args.input)},
                "reaction_map": {"path": args.reaction_map.resolve().as_posix(), "sha256": sha256_file(args.reaction_map)},
                "assays": {"path": args.assays.resolve().as_posix(), "sha256": sha256_file(args.assays)},
                "output": {"path": args.output.resolve().as_posix(), "sha256": sha256_file(args.output), "rows": int(len(frame))},
                "assay_description_coverage": coverage,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"rows={len(frame)} assay_description_coverage={coverage:.3f}")


if __name__ == "__main__":
    main()
