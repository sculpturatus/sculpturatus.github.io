"""
Descriptor computation pipeline.
Computes RDKit physicochemical descriptors + Böttcher complexity for a list of SMILES.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Optional

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Böttcher — add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from bottchscore3 import calculate_bottchscore_from_smiles
    BOTTCHER_AVAILABLE = True
except Exception as e:
    BOTTCHER_AVAILABLE = False
    print(f"[Warning] Böttcher complexity unavailable: {e}")


# ── Descriptor definitions ────────────────────────────────────────────────────

DESCRIPTOR_DEFINITIONS = {
    "MW":               ("Molecular Weight",            lambda m: Descriptors.MolWt(m)),
    "ExactMW":          ("Exact Molecular Weight",      lambda m: Descriptors.ExactMolWt(m)),
    "cLogP":            ("Calculated LogP",             lambda m: Descriptors.MolLogP(m)),
    "tPSA":             ("Topological PSA",             lambda m: Descriptors.TPSA(m)),
    "HBD":              ("H-Bond Donors",               lambda m: rdMolDescriptors.CalcNumHBD(m)),
    "HBA":              ("H-Bond Acceptors",            lambda m: rdMolDescriptors.CalcNumHBA(m)),
    "RotBonds":         ("Rotatable Bonds",             lambda m: rdMolDescriptors.CalcNumRotatableBonds(m)),
    "AromaticRings":    ("Aromatic Rings",              lambda m: rdMolDescriptors.CalcNumAromaticRings(m)),
    "Rings":            ("Total Rings",                 lambda m: rdMolDescriptors.CalcNumRings(m)),
    "HeavyAtoms":       ("Heavy Atom Count",            lambda m: m.GetNumHeavyAtoms()),
    "FractionCSP3":     ("Fraction C sp3",              lambda m: rdMolDescriptors.CalcFractionCSP3(m)),
    "Stereocenters":    ("Stereocenters",               lambda m: len(rdMolDescriptors.FindMolChiralCenters(m, includeUnassigned=True))),
    "RingFusionDeg":    ("Ring Fusion Degree",          lambda m: _ring_fusion_degree(m)),
    "BertzCT":          ("Bertz Complexity (CT)",       lambda m: Descriptors.BertzCT(m)),
    "QED":              ("Drug-likeness (QED)",         lambda m: QED.qed(m)),
    "MolRefract":       ("Molar Refractivity",          lambda m: Descriptors.MolMR(m)),
    "NumHeteroatoms":   ("Heteroatom Count",            lambda m: rdMolDescriptors.CalcNumHeteroatoms(m)),
    "NumAmideBonds":    ("Amide Bonds",                 lambda m: rdMolDescriptors.CalcNumAmideBonds(m)),
    "NumBridgeheads":   ("Bridgehead Atoms",            lambda m: rdMolDescriptors.CalcNumBridgeheadAtoms(m)),
    "NumSpiro":         ("Spiro Atoms",                 lambda m: rdMolDescriptors.CalcNumSpiroAtoms(m)),
    "MaxRingSize":      ("Max Ring Size",               lambda m: _max_ring_size(m)),
}

if BOTTCHER_AVAILABLE:
    DESCRIPTOR_DEFINITIONS["Bottcher"] = ("Böttcher Complexity", None)  # handled separately

# ── Derived / size-normalised descriptors ─────────────────────────────────────
# These are computed AFTER the base pass (need MW and Bottcher to exist in row).
# Stored as sentinel None in DESCRIPTOR_DEFINITIONS so they appear in the key
# list and are recognised by get_descriptor_label(); actual computation happens
# in compute_descriptors() below.
DESCRIPTOR_DEFINITIONS["Bottcher_per_MW"] = ("Complexity/MW",      None)   # Böttcher normalised by MW
DESCRIPTOR_DEFINITIONS["tPSA_per_MW"]     = ("tPSA/MW",            None)   # polarity density
DESCRIPTOR_DEFINITIONS["AromaticFraction"] = ("Aromatic Fraction", None)   # aromatic atoms / heavy atoms


def _ring_fusion_degree(mol):
    """Average number of bonds shared between ring systems."""
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    if len(rings) < 2:
        return 0
    shared = 0
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared += len(set(rings[i]) & set(rings[j]))
    return shared


def _max_ring_size(mol):
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    if not rings:
        return 0
    return max(len(r) for r in rings)


def _aromatic_fraction(mol) -> float:
    """Fraction of heavy atoms that are aromatic."""
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    return aromatic / heavy


def _largest_fragment_smiles(smi: str) -> str:
    """Return the SMILES of the largest fragment (strips salts/counterions)."""
    if '.' not in smi:
        return smi
    frags = smi.split('.')
    # Pick the fragment whose RDKit mol has the most heavy atoms
    best_smi, best_n = smi, 0
    for f in frags:
        mol = Chem.MolFromSmiles(f)
        if mol is not None:
            n = mol.GetNumHeavyAtoms()
            if n > best_n:
                best_n = n
                best_smi = f
    return best_smi


# ── Main computation ──────────────────────────────────────────────────────────

def compute_descriptors(names: list, smiles_list: list,
                        progress_callback=None) -> pd.DataFrame:
    """
    Compute all descriptors for a list of (name, smiles) pairs.
    Returns a DataFrame with columns: NAME, SMILES, + all descriptor keys.
    Rows with invalid SMILES are included with NaN descriptors.
    """
    records = []
    n = len(smiles_list)

    for i, (name, smi) in enumerate(zip(names, smiles_list)):
        if progress_callback:
            progress_callback(i, n, name)

        row = {"NAME": name, "SMILES": smi}
        mol = Chem.MolFromSmiles(smi) if smi else None

        for key, (label, fn) in DESCRIPTOR_DEFINITIONS.items():
            if key == "Bottcher":
                continue  # handled below
            if mol is None:
                row[key] = np.nan
            else:
                try:
                    row[key] = fn(mol)
                except Exception:
                    row[key] = np.nan

        # Böttcher — uses OpenBabel separately; strip salts/counterions first
        if BOTTCHER_AVAILABLE and smi and mol is not None:
            try:
                bottcher_smi = _largest_fragment_smiles(smi)
                row["Bottcher"] = calculate_bottchscore_from_smiles(bottcher_smi)
            except Exception:
                row["Bottcher"] = np.nan
        elif BOTTCHER_AVAILABLE:
            row["Bottcher"] = np.nan

        # ── Derived descriptors (size-normalised) ─────────────────────────────
        mw = row.get("MW", np.nan)

        # Böttcher / MW
        if BOTTCHER_AVAILABLE:
            b = row.get("Bottcher", np.nan)
            try:
                row["Bottcher_per_MW"] = float(b) / float(mw) if float(mw) > 0 else np.nan
            except Exception:
                row["Bottcher_per_MW"] = np.nan
        else:
            # Fall back to BertzCT / MW when Böttcher unavailable
            try:
                bertz = row.get("BertzCT", np.nan)
                row["Bottcher_per_MW"] = float(bertz) / float(mw) if float(mw) > 0 else np.nan
            except Exception:
                row["Bottcher_per_MW"] = np.nan

        # tPSA / MW
        try:
            tpsa = row.get("tPSA", np.nan)
            row["tPSA_per_MW"] = float(tpsa) / float(mw) if float(mw) > 0 else np.nan
        except Exception:
            row["tPSA_per_MW"] = np.nan

        # Aromatic fraction
        if mol is not None:
            try:
                row["AromaticFraction"] = _aromatic_fraction(mol)
            except Exception:
                row["AromaticFraction"] = np.nan
        else:
            row["AromaticFraction"] = np.nan

        records.append(row)

    if progress_callback:
        progress_callback(n, n, "Done")

    return pd.DataFrame(records)


def get_descriptor_columns() -> list:
    """Return list of descriptor column keys."""
    return list(DESCRIPTOR_DEFINITIONS.keys())


def get_descriptor_label(key: str) -> str:
    """Return human-readable label for a descriptor key."""
    if key in DESCRIPTOR_DEFINITIONS:
        return DESCRIPTOR_DEFINITIONS[key][0]
    return key


# ── Molecule rendering ────────────────────────────────────────────────────────

def smiles_to_png_bytes(smiles: str, width: int = 300, height: int = 250) -> Optional[bytes]:
    """Render a SMILES to PNG bytes using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None