import tempfile

import pytest

from services.workspace_prediction import predict_workspace_deployment
from services.workspace_training import train_workspace_dataset


def _records():
    smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "C1CCCCC1", "c1ccccc1", "COC", "CNC", "CC(=O)O"]
    return [
        {
            "smiles": value,
            "bioactivity_class": "Active" if index % 2 else "Inactive",
            "scaffold": f"scaffold-{index}",
            "target_identity": "P12345",
        }
        for index, value in enumerate(smiles)
    ]


def test_deployed_classification_supports_single_and_invalid_bulk_rows():
    with tempfile.TemporaryDirectory() as folder:
        trained = train_workspace_dataset(
            _records(),
            task_type="classification",
            architecture="separate_models",
            model_names=["random_forest"],
            workspace_id="00000000-0000-0000-0000-000000000001",
            artifact_root=folder,
        )
        candidate = trained["candidates"]["random_forest"]
        artifact = {
            "artifact_path": candidate["artifact_path"],
            "task_type": trained["task_type"],
            "architecture": trained["architecture"],
            "target_vocabulary": trained["target_vocabulary"],
            "model_card": {"feature_schema": trained["feature_schema"]},
        }
        results = predict_workspace_deployment(
            artifact=artifact,
            training_records=_records(),
            smiles_list=["CCO", "not-a-smiles"],
            target_identity=None,
        )
    assert results[0]["task_type"] == "classification"
    assert results[0]["prediction"] in {"Active", "Inactive"}
    assert results[0]["applicability_domain"]["status"] in {"IN_DOMAIN", "BORDERLINE", "OUT_OF_DOMAIN"}
    assert results[1]["error"] == "Invalid SMILES string"


def test_pooled_prediction_rejects_unknown_target_identity():
    with tempfile.TemporaryDirectory() as folder:
        records = [{**row, "pIC50": 5.0 + index / 10, "target_identity": "P12345" if index % 2 else "P54321"} for index, row in enumerate(_records())]
        trained = train_workspace_dataset(
            records,
            task_type="regression",
            architecture="pooled_multitarget",
            model_names=["linear_regression"],
            workspace_id="00000000-0000-0000-0000-000000000002",
            artifact_root=folder,
        )
        candidate = trained["candidates"]["linear_regression"]
        artifact = {
            "artifact_path": candidate["artifact_path"],
            "task_type": trained["task_type"],
            "architecture": trained["architecture"],
            "target_vocabulary": trained["target_vocabulary"],
            "model_card": {"feature_schema": trained["feature_schema"]},
        }
        with pytest.raises(ValueError, match="Unknown target identity"):
            predict_workspace_deployment(
                artifact=artifact,
                training_records=records,
                smiles_list=["CCO"],
                target_identity="P99999",
            )
