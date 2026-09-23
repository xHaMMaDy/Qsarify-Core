import tempfile

from services.workspace_training import train_workspace_dataset


def _records():
    smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "C1CCCCC1", "c1ccccc1", "COC", "CNC", "CC(=O)O"]
    return [{"smiles": value, "bioactivity_class": "Active" if index % 2 else "Inactive", "scaffold": f"scaffold-{index}", "target_identity": "P12345"} for index, value in enumerate(smiles)]


def test_classification_training_returns_candidates_metrics_and_ad():
    with tempfile.TemporaryDirectory() as folder:
        result = train_workspace_dataset(_records(), task_type="classification", architecture="separate_models", model_names=["random_forest", "extra_trees"], workspace_id="00000000-0000-0000-0000-000000000001", artifact_root=folder)
    assert result["best_model"] in {"random_forest", "extra_trees"}
    assert result["candidates"][result["best_model"]]["metrics"]["mcc"] <= 1.0
    assert "out_of_domain_fraction" in result["candidates"][result["best_model"]]["ad"]


def test_pooled_regression_keeps_target_vocabulary():
    records = [{**row, "pIC50": 5.0 + (index / 10), "target_identity": "P12345" if index % 2 else "P54321"} for index, row in enumerate(_records())]
    with tempfile.TemporaryDirectory() as folder:
        result = train_workspace_dataset(records, task_type="regression", architecture="pooled_multitarget", model_names=["linear_regression"], workspace_id="00000000-0000-0000-0000-000000000002", artifact_root=folder)
    assert result["target_vocabulary"] == ["P12345", "P54321"]
    assert "r2" in result["candidates"]["linear_regression"]["metrics"]


def test_training_computes_scaffolds_when_collected_rows_do_not_include_them():
    records = [{key: value for key, value in row.items() if key != "scaffold"} for row in _records()]
    with tempfile.TemporaryDirectory() as folder:
        result = train_workspace_dataset(records, task_type="classification", architecture="separate_models", model_names=["random_forest"], workspace_id="00000000-0000-0000-0000-000000000003", artifact_root=folder)
    assert result["scaffold_count"] >= 2
