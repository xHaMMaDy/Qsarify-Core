"""Prediction helpers that attach applicability-domain evidence."""

from __future__ import annotations

from .applicability_domain import assess_molecule_applicability_domain


def predict_with_applicability_domain(
    model,
    features,
    molecule,
    training_fingerprints,
    training_bit_counts=None,
):
    """Return prediction, confidence, and real-time AD evidence."""

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))
    ad = assess_molecule_applicability_domain(
        molecule,
        training_fingerprints,
        training_bit_counts,
    )
    return prediction, confidence, ad
