from argparse import Namespace
from pathlib import Path

from experiments.run_benchmark import run_benchmark
from experiments.run_benchmark import parse_smiles
from experiments.run_benchmark import get_split_indices
from experiments.run_benchmark import prepare_benchmark_frame
from experiments.run_y_randomization import permute_labels
from services.curation import curate_smiles
import numpy as np
import pandas as pd


def test_benchmark_runner_emits_reproducible_artifacts(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "benchmark_fixture.csv"
    args = Namespace(
        input=str(fixture),
        output=str(tmp_path / "benchmark"),
        models="LogisticRegression",
        fingerprint_type="Morgan",
        fingerprint_radius=2,
        fingerprint_bits=64,
        descriptors="MolWt,MolLogP,TPSA",
        one_hot_column=None,
            test_size=0.5,
        seed=42,
        split_strategy="random",
        imputer_strategy="mean",
        resampling=False,
        save_models=False,
    )

    payload = run_benchmark(args)

    output = Path(args.output)
    assert (output / "manifest.json").exists()
    assert (output / "results.json").exists()
    assert payload["manifest"]["input"]["sha256"]
    assert payload["manifest"]["feature_metadata"]["feature_count"] == 67
    assert "LogisticRegression" in payload["results"]
    assert payload["results"]["LogisticRegression"]["pr_auc"] is not None
    assert payload["results"]["LogisticRegression"]["pr_curve"] is not None
    assert "mcc" in payload["results"]["LogisticRegression"]
    assert "balanced_accuracy" in payload["results"]["LogisticRegression"]


def test_benchmark_smiles_parser_rejects_missing_and_non_string_values():
    assert parse_smiles(None) is None
    assert parse_smiles(float("nan")) is None
    assert parse_smiles("not-a-smiles") is None
    assert parse_smiles("CCO") is not None


def test_chemical_curation_strips_salts_and_neutralizes_ions():
    assert curate_smiles("CC(=O)[O-].[Na+]") == "CC(=O)O"
    assert curate_smiles("C[NH2+]C") == "CNC"
    assert curate_smiles("not-a-smiles") is None


def test_y_randomization_preserves_pooled_class_count_and_is_reproducible():
    frame = pd.DataFrame(
        {
            "target_name": ["AChE"] * 4 + ["BACE1"] * 4,
            "smiles": ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "c1ccncc1", "COC", "CNC"],
            "bioactivity_class": ["Active", "Active", "Inactive", "Inactive", "Active", "Inactive", "Inactive", "Inactive"],
        }
    )

    first = permute_labels(frame, seed=42)
    second = permute_labels(frame, seed=42)
    assert first["bioactivity_class"].tolist() == second["bioactivity_class"].tolist()
    assert first["bioactivity_class"].value_counts().to_dict() == frame["bioactivity_class"].value_counts().to_dict()


def test_scaffold_split_has_no_group_overlap():
    index = np.arange(8)
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=index)
    groups = pd.Series(["a", "a", "b", "b", "c", "c", "d", "d"], index=index)
    train, test = get_split_indices(
        index, y, test_size=0.5, seed=42, split_strategy="scaffold", groups=groups
    )
    assert set(groups.loc[train]).isdisjoint(set(groups.loc[test]))


def test_benchmark_record_cleaning_filters_invalid_activity_and_aggregates_duplicates():
    frame = pd.DataFrame(
        {
            "target_name": ["AChE", "AChE", "AChE", "AChE", "AChE"],
            "smiles": ["C(C)O", "CCO", "c1ccccc1", "c1ccccc1", "not-a-smiles"],
            "standard_units": ["nM"] * 5,
            "standard_value": [100.0, 20000.0, 500.0, 500.0, 100.0],
            "bioactivity_class": ["Active", "Inactive", "Active", "Active", "Active"],
        }
    )

    cleaned, audit = prepare_benchmark_frame(frame, one_hot_column="target_name")

    assert len(cleaned) == 2
    assert set(cleaned["smiles"]) == {"CCO", "c1ccccc1"}
    assert audit["duplicate_rows_collapsed"] == 2
    assert audit["conflicting_duplicate_groups"] == 1
    assert audit["invalid_smiles_rows"] == 1
    assert cleaned.loc[cleaned["smiles"] == "CCO", "bioactivity_class"].iloc[0] == "Inactive"


def test_target_specific_records_are_preserved_without_target_one_hot_encoding():
    frame = pd.DataFrame(
        {
            "target_name": ["AChE", "BACE1"],
            "smiles": ["CCO", "CCO"],
            "standard_units": ["nM", "nM"],
            "standard_value": [100.0, 20000.0],
            "bioactivity_class": ["Active", "Inactive"],
        }
    )

    cleaned, audit = prepare_benchmark_frame(frame, one_hot_column=None)

    assert len(cleaned) == 2
    assert audit["dedup_group_columns"] == ["target_name", "_canonical_smiles"]
