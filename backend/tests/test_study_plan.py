import services.study_plan as study_plan


def valid_plan():
    return {
        "architecture": "pooled_multitarget",
        "task_types": ["regression", "classification"],
        "endpoint_policy": {"compatible_groups": [], "incompatible_combinations": ["IC50 with Ki"], "pactivity_conversion": {"allowed": False}},
        "target_encoding": "one_hot_target_identity",
        "candidate_models": ["random_forest", "extra_trees"],
        "split_strategy": "scaffold_disjoint",
        "applicability_domain": {"method": "morgan_tanimoto"},
        "warnings": [], "rationale": ["Targets share a compatible endpoint group."],
        "estimated_runtime_minutes": 8, "confidence": 0.8, "requires_user_confirmation": True,
    }


def test_pooled_plan_requires_target_identity_encoding():
    plan = valid_plan()
    plan["target_encoding"] = "none"
    try:
        study_plan.validate_study_plan(plan, 2)
    except ValueError as exc:
        assert "encode target identity" in str(exc)
    else:
        raise AssertionError("Expected pooled target encoding validation to fail")


def test_separate_plan_forbids_pooled_target_encoding():
    plan = valid_plan()
    plan["architecture"] = "separate_models"
    try:
        study_plan.validate_study_plan(plan, 2)
    except ValueError as exc:
        assert "Separate target models" in str(exc)
    else:
        raise AssertionError("Expected separate-model encoding validation to fail")


def test_study_plan_normalizes_models_and_requires_confirmation():
    plan = valid_plan()
    plan["candidate_models"] = ["random_forest", "random_forest", "extra_trees"]
    plan["requires_user_confirmation"] = False
    normalized = study_plan.validate_study_plan(plan, 2)
    assert normalized["candidate_models"] == ["random_forest", "extra_trees"]
    assert normalized["requires_user_confirmation"] is True


def test_endpoint_policy_rejects_incompatible_grouping():
    plan = valid_plan()
    plan["endpoint_policy"]["compatible_groups"] = [{"endpoints": ["IC50", "Ki"], "units": ["nM"]}]
    try:
        study_plan.validate_study_plan(plan, 2)
    except ValueError as exc:
        assert "Incompatible endpoints" in str(exc)
    else:
        raise AssertionError("Expected incompatible endpoint grouping to fail")


def test_pactivity_conversion_requires_confirmation_and_basis():
    plan = valid_plan()
    plan["endpoint_policy"]["pactivity_conversion"] = {"requested": True, "from_endpoint": "IC50"}
    try:
        study_plan.validate_study_plan(plan, 2)
    except ValueError as exc:
        assert "explicit user confirmation" in str(exc)
    else:
        raise AssertionError("Expected unconfirmed pActivity conversion to fail")
    plan["endpoint_policy"]["pactivity_conversion"] = {"requested": True, "from_endpoint": "IC50", "to_endpoint": "pActivity", "user_confirmed": True, "scientific_basis": "Same assay units and explicitly approved log transform."}
    assert study_plan.validate_study_plan(plan, 2)["requires_user_confirmation"] is True


def test_optional_execution_metadata_is_structured_and_validated():
    plan = valid_plan()
    plan.update({
        "activity_transform": {"name": "pActivity", "confirmed": False},
        "classification_thresholds": [{"endpoint": "IC50", "unit": "nM", "threshold": 1000}],
        "feature_strategy": {"fingerprint": "Morgan", "radius": 2, "bits": 2048},
        "class_balance_strategy": {"method": "none"},
        "evaluation_metrics": ["MCC", "balanced_accuracy", "ROC-AUC"],
        "estimated_records": {"per_target": {"P05067": 120}},
    })
    normalized = study_plan.validate_study_plan(plan, 2)
    assert normalized["feature_strategy"]["fingerprint"] == "Morgan"
    assert normalized["evaluation_metrics"] == ["MCC", "balanced_accuracy", "ROC-AUC"]


def test_both_architecture_requires_one_hot_identity_for_pooled_branch():
    plan = valid_plan()
    plan["architecture"] = "both"
    plan["target_encoding"] = "none"
    try:
        study_plan.validate_study_plan(plan, 2)
    except ValueError as exc:
        assert "encode target identity" in str(exc)
    else:
        raise AssertionError("Expected both architecture to validate its pooled branch")

    plan["target_encoding"] = "one_hot_target_identity"
    assert study_plan.validate_study_plan(plan, 2)["architecture"] == "both"


def test_study_plan_uses_configured_reasoning_model(monkeypatch):
    seen = {}

    def fake_call(model, _system, _user, _schema, **_kwargs):
        seen["model"] = model
        return valid_plan(), {"provider": "fixture", "model": model, "usage": {}}

    monkeypatch.setenv("OPENROUTER_REASONING_MODEL", "fixture/strong-planner")
    monkeypatch.delenv("OPENROUTER_PLAN_MODEL", raising=False)
    monkeypatch.setattr(study_plan, "call_structured", fake_call)
    study_plan.generate_study_plan([{"target_id": "P12345", "canonical_name": "Fixture target"}])
    assert seen["model"] == "fixture/strong-planner"
