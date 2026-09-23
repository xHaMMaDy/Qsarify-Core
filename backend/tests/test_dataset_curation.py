import services.dataset_curation as dataset_curation


def test_dataset_curation_preserves_real_rows_and_excludes_invalid(monkeypatch):
    values = {"CCO": "CCO", "not-a-smiles": None}
    monkeypatch.setattr(dataset_curation, "curate_smiles", lambda value: values.get(value))
    records, summary = dataset_curation.curate_dataset([
        {"molecule_chembl_id": "CHEMBL1", "smiles": "CCO", "standard_value": 10},
        {"molecule_chembl_id": "CHEMBL2", "smiles": "not-a-smiles", "standard_value": 20},
        {"molecule_chembl_id": "CHEMBL3", "smiles": None, "standard_value": 30},
    ])
    assert len(records) == 1
    assert records[0]["curated_smiles"] == "CCO"
    assert summary == {"input_records": 3, "curated_records": 1, "excluded_invalid": 1, "excluded_missing_smiles": 1}
