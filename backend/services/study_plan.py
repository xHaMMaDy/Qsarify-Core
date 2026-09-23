"""Structured AI QSAR study-plan generation and deterministic validation."""

from __future__ import annotations

import os
from typing import Any, Mapping

from services.llm_provider import ProviderError, call_structured

ALLOWED_ARCHITECTURES = {"separate_models", "pooled_multitarget", "both", "user_choice_required"}
ALLOWED_TASKS = {"regression", "classification"}
ALLOWED_MODELS = {"linear_regression", "logistic_regression", "random_forest", "extra_trees", "hist_gradient_boosting", "xgboost", "support_vector_machine", "neural_network"}
MODEL_ALIASES = {"gradient_boosting": "hist_gradient_boosting"}
INCOMPATIBLE_ENDPOINTS = {
    frozenset({"ic50", "ki"}), frozenset({"ic50", "kd"}), frozenset({"ic50", "ec50"}),
    frozenset({"ki", "kd"}), frozenset({"ki", "ec50"}), frozenset({"kd", "ec50"}),
}

STUDY_PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["architecture", "task_types", "endpoint_policy", "target_encoding", "candidate_models", "split_strategy", "applicability_domain", "warnings", "rationale", "estimated_runtime_minutes", "confidence", "requires_user_confirmation"],
    "properties": {
        "architecture": {"type": "string", "enum": sorted(ALLOWED_ARCHITECTURES)},
        "task_types": {"type": "array", "items": {"type": "string", "enum": sorted(ALLOWED_TASKS)}, "minItems": 1, "uniqueItems": True},
        "endpoint_policy": {"type": "object", "additionalProperties": False, "required": ["compatible_groups", "incompatible_combinations", "pactivity_conversion"], "properties": {
            "compatible_groups": {"type": "array", "items": {"type": "object"}},
            "incompatible_combinations": {"type": "array", "items": {"type": "string"}},
            "pactivity_conversion": {"type": "object"},
        }},
        "activity_transform": {"type": "object"},
        "classification_thresholds": {"type": "array", "items": {"type": "object"}},
        "feature_strategy": {"type": "object"},
        "class_balance_strategy": {"type": "object"},
        "evaluation_metrics": {"type": "array", "items": {"type": "string"}},
        "estimated_records": {"type": "object"},
        "target_encoding": {"type": "string", "enum": ["none", "one_hot_target_identity"]},
        "candidate_models": {"type": "array", "items": {"type": "string", "enum": sorted(ALLOWED_MODELS | set(MODEL_ALIASES))}, "minItems": 1},
        "split_strategy": {"type": "string", "enum": ["scaffold_disjoint"]},
        "applicability_domain": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "array", "items": {"type": "string"}},
        "estimated_runtime_minutes": {"type": "number", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "requires_user_confirmation": {"type": "boolean"},
    },
}


def validate_study_plan(plan: Mapping[str, Any], target_count: int) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise ValueError("Study plan must be an object")
    architecture = str(plan.get("architecture") or "")
    if architecture not in ALLOWED_ARCHITECTURES:
        raise ValueError("Study plan contains an unsupported modeling architecture")
    tasks = plan.get("task_types")
    if not isinstance(tasks, list) or not tasks or any(str(task) not in ALLOWED_TASKS for task in tasks):
        raise ValueError("Study plan must select regression and/or classification")
    encoding = str(plan.get("target_encoding") or "")
    if architecture in {"pooled_multitarget", "both"} and target_count > 1 and encoding != "one_hot_target_identity":
        raise ValueError("Pooled multi-target plans must encode target identity")
    if architecture == "separate_models" and encoding != "none":
        raise ValueError("Separate target models cannot use pooled target identity encoding")
    if plan.get("split_strategy") != "scaffold_disjoint":
        raise ValueError("Scaffold-disjoint splitting is required")
    _validate_endpoint_policy(plan.get("endpoint_policy"))
    for field in ("activity_transform", "feature_strategy", "class_balance_strategy", "estimated_records"):
        if field in plan and not isinstance(plan.get(field), Mapping):
            raise ValueError(f"{field} must be an object when supplied")
    for field in ("classification_thresholds",):
        if field in plan and not isinstance(plan.get(field), list):
            raise ValueError(f"{field} must be an array when supplied")
    if "evaluation_metrics" in plan:
        metrics = plan.get("evaluation_metrics")
        if not isinstance(metrics, list) or any(not isinstance(metric, str) or not metric.strip() for metric in metrics):
            raise ValueError("evaluation_metrics must contain non-empty names")
    models = plan.get("candidate_models")
    if not isinstance(models, list) or not models:
        raise ValueError("Study plan contains an unsupported model family")
    normalized_models = []
    model_alias_warnings = []
    for model in models:
        raw_model = str(model)
        canonical_model = MODEL_ALIASES.get(raw_model, raw_model)
        if canonical_model not in ALLOWED_MODELS:
            raise ValueError("Study plan contains an unsupported model family")
        normalized_models.append(canonical_model)
        if raw_model != canonical_model:
            model_alias_warnings.append(f"Model family '{raw_model}' was normalized to '{canonical_model}' by the deterministic trainer.")
    if target_count < 1:
        raise ValueError("At least one selected target is required")
    normalized = dict(plan)
    normalized["candidate_models"] = list(dict.fromkeys(normalized_models))
    if model_alias_warnings:
        normalized["warnings"] = list(dict.fromkeys([*(normalized.get("warnings") or []), *model_alias_warnings]))
    normalized["task_types"] = list(dict.fromkeys(str(task) for task in tasks))
    normalized["requires_user_confirmation"] = True
    return normalized


def _endpoint_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "").replace("_", "")
    for candidate in ("ic50", "ki", "kd", "ec50"):
        if candidate in text:
            return candidate
    return text


def _validate_endpoint_policy(policy: Any) -> None:
    """Reject scientifically incompatible grouping/conversion recommendations."""
    if not isinstance(policy, Mapping):
        raise ValueError("endpoint_policy is required")
    groups = policy.get("compatible_groups")
    if not isinstance(groups, list):
        raise ValueError("endpoint_policy.compatible_groups must be an array")
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("Each endpoint compatibility group must be an object")
        raw_endpoints = group.get("endpoints") or group.get("endpoint_names") or []
        if not isinstance(raw_endpoints, list):
            raise ValueError("Endpoint group endpoints must be an array")
        endpoints = {_endpoint_name(value) for value in raw_endpoints if str(value).strip()}
        for pair in INCOMPATIBLE_ENDPOINTS:
            if pair.issubset(endpoints):
                raise ValueError(f"Incompatible endpoints cannot share a group: {sorted(pair)}")
        units = group.get("units")
        if units is not None and not isinstance(units, list):
            raise ValueError("Endpoint group units must be an array when supplied")
        if isinstance(units, list) and len({str(unit).strip().lower() for unit in units if str(unit).strip()}) > 1:
            raise ValueError("Endpoint groups with multiple units require deterministic normalization before pooling")
    conversion = policy.get("pactivity_conversion") or {}
    if not isinstance(conversion, Mapping):
        raise ValueError("pactivity_conversion must be an object")
    requested = bool(conversion.get("requested") or conversion.get("enabled") or conversion.get("apply"))
    if requested:
        if conversion.get("user_confirmed") is not True:
            raise ValueError("pActivity conversion requires explicit user confirmation")
        if not str(conversion.get("scientific_basis") or "").strip():
            raise ValueError("pActivity conversion requires a scientific basis")
        source = _endpoint_name(conversion.get("from_endpoint"))
        target = _endpoint_name(conversion.get("to_endpoint") or "pactivity")
        if not source or target != "pactivity":
            raise ValueError("pActivity conversion must specify a source endpoint and pActivity target")


def generate_study_plan(targets: list[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not targets or len(targets) > 20:
        raise ValueError("Study plan requires 1-20 selected targets")
    dossier = {"targets": targets, "target_count": len(targets), "scientific_constraints": {
        "do_not_mix_incompatible_endpoints": True,
        "ic50_ki_kd_ec50_not_interchangeable": True,
        "pactivity_conversion_requires_user_confirmation": True,
        "regression_and_classification_are_separate": True,
        "scaffold_disjoint_split_required": True,
        "ensembles_disabled_by_default": True,
    }}
    system = (
        "You are QSARify's scientific QSAR planning assistant. Create a structured study-plan recommendation from the "
        "target dossier. Never fabricate data, counts, endpoints, citations, metrics, costs, or runtime. Keep separate "
        "and pooled multi-target modeling distinct. Never mix IC50, Ki, Kd, EC50, or other endpoints unless the dossier "
        "proves compatibility. Recommend one-hot target identity only for pooled models. Regression and classification "
        "are separate tasks. Use scaffold_disjoint splitting. Compare individual model families only; do not recommend "
        "ensembles by default. The user must review and confirm the plan before any data retrieval or training. Candidate model names must be chosen only from: linear_regression, logistic_regression, random_forest, extra_trees, hist_gradient_boosting, xgboost, support_vector_machine, neural_network. Include endpoint names and explicit unit normalization when a group contains multiple units. Return only the required JSON schema."
    )
    try:
        # Planning is a consequential recommendation, so prefer the configured
        # reasoning route while retaining the cost-efficient model as a safe
        # local fallback when no stronger route is configured.
        model = os.environ.get("OPENROUTER_PLAN_MODEL") or os.environ.get("OPENROUTER_REASONING_MODEL") or "google/gemini-2.5-flash-lite"
        result, metadata = call_structured(model, system, str(dossier), STUDY_PLAN_SCHEMA, max_tokens=1800)
    except ProviderError:
        raise
    return validate_study_plan(result, len(targets)), metadata
