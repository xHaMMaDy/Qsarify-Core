"""Morgan/Tanimoto applicability-domain calculations."""

from __future__ import annotations

import numpy as np
from rdkit.Chem import AllChem


AD_STATUS_IN_DOMAIN = "IN_DOMAIN"
AD_STATUS_BORDERLINE = "BORDERLINE"
AD_STATUS_OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
AD_STATUS_UNAVAILABLE = "UNAVAILABLE"


def pack_fingerprint_matrix(fingerprint_matrix) -> np.ndarray:
    """Pack a binary fingerprint matrix into compact uint8 rows."""

    matrix = np.asarray(fingerprint_matrix, dtype=np.uint8)
    if matrix.ndim != 2:
        raise ValueError("Fingerprint matrix must be two-dimensional")
    return np.packbits(matrix, axis=1, bitorder="big")


def unpacked_bit_counts(packed_fingerprints: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed_fingerprints, dtype=np.uint8)
    if packed.ndim != 2:
        raise ValueError("Packed fingerprints must be two-dimensional")
    return np.unpackbits(packed, axis=1, bitorder="big").sum(axis=1).astype(np.int32)


def max_tanimoto_similarity(query_fingerprint, packed_training_fingerprints, training_bit_counts=None) -> float:
    """Return the maximum Tanimoto similarity to packed training fingerprints."""

    training = np.asarray(packed_training_fingerprints, dtype=np.uint8)
    if training.ndim != 2 or training.shape[0] == 0:
        raise ValueError("Training fingerprints are unavailable")
    query = np.asarray(query_fingerprint, dtype=np.uint8).reshape(-1)
    query_packed = np.packbits(query, bitorder="big")
    if query_packed.shape[0] != training.shape[1]:
        raise ValueError("Query and training fingerprint widths do not match")
    query_count = int(query.sum())
    counts = (
        np.asarray(training_bit_counts, dtype=np.int32)
        if training_bit_counts is not None
        else unpacked_bit_counts(training)
    )
    intersections = np.unpackbits(np.bitwise_and(training, query_packed), axis=1, bitorder="big").sum(axis=1)
    unions = counts + query_count - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions > 0,
    )
    return float(np.max(similarities))


def similarity_status(max_similarity: float) -> str:
    if max_similarity >= 0.50:
        return AD_STATUS_IN_DOMAIN
    if max_similarity >= 0.30:
        return AD_STATUS_BORDERLINE
    return AD_STATUS_OUT_OF_DOMAIN


def assess_molecule_applicability_domain(
    molecule,
    packed_training_fingerprints,
    training_bit_counts=None,
    *,
    radius: int = 3,
    n_bits: int = 2048,
) -> dict:
    """Compute the real-time maximum Morgan/Tanimoto AD status."""

    if molecule is None or packed_training_fingerprints is None:
        return {"max_similarity": None, "status": AD_STATUS_UNAVAILABLE}
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)
    query_bits = np.asarray(fingerprint, dtype=np.uint8)
    maximum = max_tanimoto_similarity(query_bits, packed_training_fingerprints, training_bit_counts)
    return {"max_similarity": maximum, "status": similarity_status(maximum)}
