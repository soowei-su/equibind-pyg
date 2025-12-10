"""
Data preprocessing and featurization for PDBBind complexes.

This module contains all logic for parsing raw PDBBind files (`.pdb`,
`.sdf`) using RDKit and converting them into the feature tensors
(`x`, `pos`, `edge_index`) expected by `PDBBindPyG`.

All functions are designed to be called by `pdbbind_dataset.py`. Any
parsing or featurization failure is handled by returning `None`, so the
dataset class can skip problematic complexes gracefully.
"""

import logging
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -------------------------------------------------------------------------
# Feature definitions
# -------------------------------------------------------------------------

STANDARD_RESIDUES = [
    "ALA", "CYS", "ASP", "GLU", "PHE",
    "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG",
    "SER", "THR", "VAL", "TRP", "TYR",
]
RESIDUE_FEATURE_DIM = len(STANDARD_RESIDUES) + 1  # +1 for "other/unknown"
assert RESIDUE_FEATURE_DIM == 21

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B"]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
ATOM_DEGREES = [0, 1, 2, 3, 4, 5, 6]

# Total ligand feature dimension:
# (len(ATOM_SYMBOLS) + 1) + (len(HYBRIDIZATION_TYPES) + 1) +
# (len(ATOM_DEGREES) + 1) + 3 scalar features
LIGAND_ATOM_FEATURE_DIM = (
    (len(ATOM_SYMBOLS) + 1)
    + (len(HYBRIDIZATION_TYPES) + 1)
    + (len(ATOM_DEGREES) + 1)
    + 3  # formal charge, aromatic flag, in_ring flag
)
assert LIGAND_ATOM_FEATURE_DIM == 28

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------


def one_hot_encode(value, allowed_set, as_tensor: bool = False):
    """Return one-hot encoding for `value` against an allowed set + 'other'."""
    encoding = [0.0] * (len(allowed_set) + 1)
    try:
        idx = allowed_set.index(value)
        encoding[idx] = 1.0
    except ValueError:
        encoding[-1] = 1.0

    if as_tensor:
        return torch.tensor(encoding, dtype=torch.float)
    return encoding


def get_atom_features(atom: Chem.Atom) -> torch.Tensor:
    """
    Generate a 28-dim feature vector for a single ligand atom.

    Parameters
    ----------
    atom : rdkit.Chem.Atom
        RDKit atom object.

    Returns
    -------
    torch.Tensor
        Tensor of shape [LIGAND_ATOM_FEATURE_DIM].
    """
    f_symbol = one_hot_encode(atom.GetSymbol(), ATOM_SYMBOLS, as_tensor=True)
    f_hybrid = one_hot_encode(atom.GetHybridization(), HYBRIDIZATION_TYPES, as_tensor=True)
    f_degree = one_hot_encode(atom.GetDegree(), ATOM_DEGREES, as_tensor=True)

    f_scalar = torch.tensor(
        [
            atom.GetFormalCharge(),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ],
        dtype=torch.float,
    )

    features = torch.cat([f_symbol, f_hybrid, f_degree, f_scalar], dim=0)
    assert features.shape[0] == LIGAND_ATOM_FEATURE_DIM, (
        f"Atom feature vector has length {features.shape[0]}, "
        f"expected {LIGAND_ATOM_FEATURE_DIM}"
    )
    return features


def get_residue_features(res_name: str) -> torch.Tensor:
    """
    Generate a residue-level feature vector.

    Parameters
    ----------
    res_name : str
        Three-letter residue name (e.g. 'ALA').

    Returns
    -------
    torch.Tensor
        Tensor of shape [RESIDUE_FEATURE_DIM].
    """
    features = one_hot_encode(res_name, STANDARD_RESIDUES, as_tensor=True)
    assert features.shape[0] == RESIDUE_FEATURE_DIM, (
        f"Residue feature vector has length {features.shape[0]}, "
        f"expected {RESIDUE_FEATURE_DIM}"
    )
    return features


# -------------------------------------------------------------------------
# Main parsing functions
# -------------------------------------------------------------------------


def parse_ligand(ligand_file: Path):
    """
    Parse a ligand `.sdf` file into node features, coordinates, and bonds.

    Parameters
    ----------
    ligand_file : Path
        Path to the ligand SDF file.

    Returns
    -------
    dict or None
        A dictionary with keys `x`, `pos`, `edge_index`, `mol` on success,
        or None if parsing fails.
    """
    try:
        supplier = Chem.SDMolSupplier(str(ligand_file), removeHs=False, sanitize=True)
        mol = supplier[0]
        if mol is None:
            logging.warning("RDKit failed to parse ligand: %s", ligand_file)
            return None

        mol = Chem.AddHs(mol, addCoords=True)
        if mol is None:
            logging.warning("RDKit failed to add Hs to ligand: %s", ligand_file)
            return None

        atom_features_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.stack(atom_features_list, dim=0)

        conformer = mol.GetConformer()
        pos = torch.tensor(conformer.GetPositions(), dtype=torch.float)

        bonds = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bonds.append([i, j])
            bonds.append([j, i])

        if not bonds:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()

        assert x.shape[0] == pos.shape[0], (
            "Ligand feature and position tensors must have the same number of atoms."
        )

        return {"x": x, "pos": pos, "edge_index": edge_index, "mol": mol}

    except Exception as e:
        logging.error("Error parsing ligand file %s: %s", ligand_file, e, exc_info=True)
        return None


def parse_receptor(receptor_file: Path):
    """
    Parse a receptor `.pdb` file into residue-level features and Cα coordinates.

    Parameters
    ----------
    receptor_file : Path
        Path to the receptor PDB file.

    Returns
    -------
    dict or None
        A dictionary with keys `x`, `pos`, `mol` on success,
        or None if parsing fails.
    """
    try:
        mol = Chem.MolFromPDBFile(str(receptor_file), removeHs=False, sanitize=True)
        if mol is None:
            logging.warning("RDKit failed to parse receptor: %s", receptor_file)
            return None

        mol = Chem.AddHs(mol, addCoords=True)
        if mol is None:
            logging.warning("RDKit failed to add Hs to receptor: %s", receptor_file)
            return None

        residues = {}
        conformer = mol.GetConformer()

        for atom in mol.GetAtoms():
            pdb_info = atom.GetPDBResidueInfo()
            if pdb_info is None:
                continue

            atom_name = pdb_info.GetName().strip()
            res_name = pdb_info.GetResidueName().strip()
            res_num = pdb_info.GetResidueNumber()
            chain_id = pdb_info.GetChainId()
            res_key = (chain_id, res_num, res_name)

            if atom_name == "CA":
                if res_key not in residues:
                    residues[res_key] = {}

                atom_idx = atom.GetIdx()
                residues[res_key]["pos"] = list(conformer.GetAtomPosition(atom_idx))
                residues[res_key]["x"] = get_residue_features(res_name)

        sorted_res_keys = sorted(residues.keys())

        res_features_list = []
        res_pos_list = []

        for res_key in sorted_res_keys:
            res_data = residues[res_key]
            if "pos" in res_data and "x" in res_data:
                res_features_list.append(res_data["x"])
                res_pos_list.append(res_data["pos"])

        if not res_features_list:
            logging.warning("No C-alpha atoms found in receptor: %s", receptor_file)
            return None

        x = torch.stack(res_features_list, dim=0)
        pos = torch.tensor(res_pos_list, dtype=torch.float)

        return {"x": x, "pos": pos, "mol": mol}

    except Exception as e:
        logging.error("Error parsing receptor file %s: %s", receptor_file, e, exc_info=True)
        return None


def process_complex(pdb_id: str, data_dir: Path):
    """
    Parse and featurize a single PDBBind complex.

    Parameters
    ----------
    pdb_id : str
        Four-character PDB ID (e.g. "1a0q").
    data_dir : Path
        Directory containing the raw `v2020/` folder.

    Returns
    -------
    dict or None
        A nested dictionary with `ligand` and `receptor` entries, or None if
        either component fails.
    """
    complex_dir = data_dir / "v2020" / pdb_id
    ligand_file = complex_dir / f"{pdb_id}_ligand.sdf"
    receptor_file = complex_dir / f"{pdb_id}_protein.pdb"

    if not ligand_file.exists():
        logging.warning("Ligand file not found: %s", ligand_file)
        return None
    if not receptor_file.exists():
        logging.warning("Receptor file not found: %s", receptor_file)
        return None

    ligand_data = parse_ligand(ligand_file)
    if ligand_data is None:
        logging.warning("Failed to process ligand for %s", pdb_id)
        return None

    receptor_data = parse_receptor(receptor_file)
    if receptor_data is None:
        logging.warning("Failed to process receptor for %s", pdb_id)
        return None

    return {"ligand": ligand_data, "receptor": receptor_data}
