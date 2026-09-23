import app as backend_app


class _Response:
    def json(self):
        return {"molecules": [{"molecule_chembl_id": "CHEMBL1", "molecule_structures": {"canonical_smiles": "CCO"}}]}


def test_molecule_batch_uses_bounded_real_api_query(monkeypatch):
    seen = {}

    def fake_get(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(backend_app, "_get_external_response", fake_get)
    result = backend_app.fetch_molecule_batch(["CHEMBL1", "CHEMBL2"])
    assert "molecule_chembl_id__in=CHEMBL1%2CCHEMBL2" in seen["url"]
    assert "limit=2" in seen["url"]
    assert result["CHEMBL1"]["molecule_structures"]["canonical_smiles"] == "CCO"
