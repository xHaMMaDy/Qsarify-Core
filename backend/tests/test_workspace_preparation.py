import services.workspace_preparation as preparation


def test_classification_preparation_is_scaffold_aware_and_counts_labels():
    result = preparation.prepare_workspace_dataset([
        {"smiles": "CCO", "bioactivity_class": "Active", "molecule_chembl_id": "CHEMBL1"},
        {"smiles": "CCN", "bioactivity_class": "Inactive", "molecule_chembl_id": "CHEMBL2"},
    ], task_type="classification")
    assert result["summary"]["task_type"] == "classification"
    assert result["summary"]["split_strategy"] == "scaffold_disjoint"
    assert result["summary"]["label_counts"] == {"1": 1, "0": 1}
    assert result["summary"]["feature_width"] == 2053


def test_pooled_preparation_records_target_identity():
    result = preparation.prepare_workspace_dataset([
        {"smiles": "CCO", "pIC50": 6.2, "target_identity": "P12345"},
        {"smiles": "CCN", "pIC50": 5.1, "target_identity": "P54321"},
    ], task_type="regression", pooled=True)
    assert result["summary"]["pooled"] is True
    assert result["summary"]["target_counts"] == {"P12345": 1, "P54321": 1}
    assert all(row["target_identity"] for row in result["prepared_records"])
