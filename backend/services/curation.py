"""Deterministic chemical structure curation for QSAR inputs."""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


ALLOWED_ATOMIC_NUMBERS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
TAUTOMER_MAX_TAUTOMERS = 32
TAUTOMER_MAX_TRANSFORMS = 32
TAUTOMER_TRIGGERS = (
    Chem.MolFromSmarts("[C,c]=[O,N,S;v2]"),
    Chem.MolFromSmarts("[nH]"),
)


def _warn(logger, reason: str) -> None:
    if logger is not None:
        logger.warning("Chemical curation excluded a structure: %s", reason)


def curate_molecule(value: str, *, logger=None):
    """Return a curated RDKit molecule or ``None`` for an unusable structure.

    The operation is deliberately fail-closed: malformed SMILES, invalid
    valence, unsupported elements, and standardization failures are excluded
    rather than silently passed to a QSAR model.
    """

    if not isinstance(value, str) or not value.strip():
        _warn(logger, "empty or non-string SMILES")
        return None
    log_block = rdBase.BlockLogs()
    log_block.__enter__()
    try:
        molecule = Chem.MolFromSmiles(value.strip())
        if molecule is None:
            _warn(logger, "SMILES parsing failed")
            return None
        Chem.SanitizeMol(molecule)

        # Keep the largest organic fragment, removing counterions and salts.
        molecule = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(molecule)
        if molecule is None or molecule.GetNumAtoms() == 0:
            _warn(logger, "no parent fragment remained after salt stripping")
            return None

        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        # Molecules without a carbonyl/imine/thione or aromatic [nH] motif do
        # not have a common tautomeric trigger; skip the expensive enumerator
        # for those structures. Pathological tautomer spaces are bounded.
        if any(trigger is not None and molecule.HasSubstructMatch(trigger) for trigger in TAUTOMER_TRIGGERS):
            tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
            tautomer_enumerator.SetMaxTautomers(TAUTOMER_MAX_TAUTOMERS)
            tautomer_enumerator.SetMaxTransforms(TAUTOMER_MAX_TRANSFORMS)
            molecule = tautomer_enumerator.Canonicalize(molecule)
        Chem.SanitizeMol(molecule)

        atomic_numbers = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
        if not atomic_numbers or not atomic_numbers.issubset(ALLOWED_ATOMIC_NUMBERS):
            _warn(logger, "unsupported element or organometallic structure")
            return None
        return molecule
    except Exception as exc:  # RDKit can raise on malformed organometallics.
        _warn(logger, f"standardization failed ({type(exc).__name__})")
        return None
    finally:
        log_block.__exit__(None, None, None)


@lru_cache(maxsize=100_000)
def _curate_smiles_cached(value: str) -> str | None:
    molecule = curate_molecule(value)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None


def curate_smiles(value: str, *, logger=None) -> str | None:
    """Return a clean canonical SMILES or ``None`` when curation fails."""

    if not isinstance(value, str) or not value.strip():
        _warn(logger, "empty or non-string SMILES")
        return None
    curated = _curate_smiles_cached(value.strip())
    if curated is None:
        _warn(logger, "standardization failed or structure was excluded")
    return curated
