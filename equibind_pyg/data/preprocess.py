"""
Data preprocessing and featurization module.

This file contains all the logic for parsing raw PDBBind files (.pdb, .sdf)
using RDKit and converting them into feature tensors (`x`, `pos`, `edge_index`)
as defined in the `v1_rigid_spec.md` schema.

All functions are designed to be called by `pdbbind_dataset.py`.
Parsing failures are handled by returning `None`, allowing the dataset
class to skip problematic complexes.
"""

import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Definitions ---

# Define the standard 20 amino acids for one-hot encoding
STANDARD_RESIDUES = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
]
# Total dimension is 20 + 1 for "other/unknown" residue type
RESIDUE_FEATURE_DIM = len(STANDARD_RESIDUES) + 1 
assert RESIDUE_FEATURE_DIM == 21

# Define atomic features for ligands
ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B'] # 10 + 1 "other"
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP, 
    Chem.rdchem.HybridizationType.SP2, 
    Chem.rdchem.HybridizationType.SP3, 
    Chem.rdchem.HybridizationType.SP3D, 
    Chem.rdchem.HybridizationType.SP3D2
] # 5 + 1 "other"
ATOM_DEGREES = [0, 1, 2, 3, 4, 5, 6] # 7 + 1 "other"

# Calculate total ligand feature dimension
LIGAND_ATOM_FEATURE_DIM = (len(ATOM_SYMBOLS) + 1) + \
                          (len(HYBRIDIZATION_TYPES) + 1) + \
                          (len(ATOM_DEGREES) + 1) + \
                          1 + 1 + 1 # charge, aromatic, in_ring
assert LIGAND_ATOM_FEATURE_DIM == 28

# --- Helper Functions ---

def one_hot_encode(value, allowed_set, as_tensor=False):
    """Creates a one-hot encoding for a value in an allowed set."""
    encoding = [0.0] * (len(allowed_set) + 1)
    try:
        idx = allowed_set.index(value)
        encoding[idx] = 1.0
    except ValueError:
        encoding[-1] = 1.0 # "Other" category
    
    if as_tensor:
        return torch.tensor(encoding, dtype=torch.float)
    return encoding

def get_atom_features(atom):
    """
    Generates a feature vector for a single ligand atom.

    Args:
        atom (rdkit.Chem.Atom): The RDKit atom object.

    Returns:
        torch.Tensor: A 1D tensor of shape [LIGAND_ATOM_FEATURE_DIM].
    """
    # 1. One-hot atomic symbol
    f_symbol = one_hot_encode(atom.GetSymbol(), ATOM_SYMBOLS, as_tensor=True) # [11]
    
    # 2. One-hot hybridization
    f_hybrid = one_hot_encode(atom.GetHybridization(), HYBRIDIZATION_TYPES, as_tensor=True) # [6]
    
    # 3. One-hot degree
    f_degree = one_hot_encode(atom.GetDegree(), ATOM_DEGREES, as_tensor=True) # [8]
    
    # 4. Scalar features
    f_scalar = torch.tensor([
        atom.GetFormalCharge(),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing())
    ], dtype=torch.float) # [3]
    
    # Concatenate all features
    features = torch.cat([f_symbol, f_hybrid, f_degree, f_scalar], dim=0) # [11 + 6 + 8 + 3 = 28]
    
    # This assertion will fail the script if any atom is not 28 features
    assert features.shape[0] == LIGAND_ATOM_FEATURE_DIM, \
        f"Atom feature vector has length {features.shape[0]}, expected {LIGAND_ATOM_FEATURE_DIM}"
    
    return features

def get_residue_features(res_name):
    """
    Generates a one-hot feature vector for a single receptor residue.
    
    Args:
        res_name (str): The 3-letter residue name (e.g., 'ALA').
    
    Returns:
        torch.Tensor: A 1D tensor of shape [RESIDUE_FEATURE_DIM].
    """
    features = one_hot_encode(res_name, STANDARD_RESIDUES, as_tensor=True)
    assert features.shape[0] == RESIDUE_FEATURE_DIM, \
        f"Residue feature vector has length {features.shape[0]}, expected {RESIDUE_FEATURE_DIM}"
    return features

# --- Main Parsing Functions ---

def parse_ligand(ligand_file: Path):
    """
    Parses an .sdf ligand file, extracts features, and returns a dictionary.
    
    Args:
        ligand_file (Path): Path to the .sdf file.
        
    Returns:
        dict or None: A dictionary containing 'x', 'pos', 'edge_index', 'mol'
                      or None if parsing fails.
    """
    try:
        # Load the molecule, keeping hydrogens (removeHs=False)
        supplier = Chem.SDMolSupplier(str(ligand_file), removeHs=False, sanitize=True)
        mol = supplier[0]
        if mol is None:
            logging.warning(f"RDKit failed to parse ligand: {ligand_file}")
            return None
            
        # Re-add hydrogens to ensure consistency and get their coordinates
        mol = Chem.AddHs(mol, addCoords=True)
        if mol is None:
            logging.warning(f"RDKit failed to add Hs to ligand: {ligand_file}")
            return None

        # 1. Get Atom Features ('x')
        atom_features_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.stack(atom_features_list, dim=0)

        # 2. Get Atom Coordinates ('pos')
        conformer = mol.GetConformer()
        pos = torch.tensor(conformer.GetPositions(), dtype=torch.float)

        # 3. Get Bonds ('edge_index')
        bonds = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Add undirected edges
            bonds.append([i, j])
            bonds.append([j, i])
            
        if not bonds:
            # Handle single-atom "molecules"
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()
            
        assert x.shape[0] == pos.shape[0], "Feature and position tensors have different number of atoms"

        return {
            'x': x,
            'pos': pos,
            'edge_index': edge_index,
            'mol': mol # Keep the RDKit mol object for potential future use
        }

    except Exception as e:
        logging.error(f"Error parsing ligand file {ligand_file}: {e}", exc_info=True)
        return None


def parse_receptor(receptor_file: Path):
    """
    Parses a .pdb receptor file, extracts residue-level features,
    and returns a dictionary.
    
    The graph is coarse-grained at the residue level, using C-Alpha
    atoms for position and feature.
    
    Args:
        receptor_file (Path): Path to the .pdb file.
        
    Returns:
        dict or None: A dictionary containing 'x', 'pos', 'mol'
                      or None if parsing fails.
    """
    try:
        mol = Chem.MolFromPDBFile(str(receptor_file), removeHs=False, sanitize=True)
        if mol is None:
            logging.warning(f"RDKit failed to parse receptor: {receptor_file}")
            return None
        
        # Add hydrogens for consistency, though we only use C-Alpha
        mol = Chem.AddHs(mol, addCoords=True)
        if mol is None:
            logging.warning(f"RDKit failed to add Hs to receptor: {receptor_file}")
            return None

        residues = {} # Use a dict to store data as we find C-Alphas
        conformer = mol.GetConformer()

        # Iterate over all atoms to find the C-Alpha of each residue
        for atom in mol.GetAtoms():
            pdb_info = atom.GetPDBResidueInfo()
            if pdb_info is None:
                continue # Skip atoms with no PDB info
                
            atom_name = pdb_info.GetName().strip()
            res_name = pdb_info.GetResidueName().strip()
            res_num = pdb_info.GetResidueNumber()
            chain_id = pdb_info.GetChainId()
            
            # Create a unique key for each residue
            res_key = (chain_id, res_num, res_name)
            
            # We only care about the C-Alpha atom
            if atom_name == 'CA':
                if res_key not in residues:
                    residues[res_key] = {}
                
                atom_idx = atom.GetIdx()
                # Get the 3D position of the C-Alpha
                residues[res_key]['pos'] = list(conformer.GetAtomPosition(atom_idx))
                # Get the one-hot feature for the residue
                residues[res_key]['x'] = get_residue_features(res_name)

        # Sort the residues to ensure a consistent node order
        sorted_res_keys = sorted(residues.keys())
        
        res_features_list = []
        res_pos_list = []
        
        for res_key in sorted_res_keys:
            res_data = residues[res_key]
            # Ensure we found both 'pos' and 'x' (i.e., a valid C-Alpha)
            if 'pos' in res_data and 'x' in res_data:
                res_features_list.append(res_data['x'])
                res_pos_list.append(res_data['pos'])
        
        if not res_features_list:
            logging.warning(f"No C-Alpha atoms found in receptor: {receptor_file}")
            return None

        # Stack the lists into tensors
        x = torch.stack(res_features_list, dim=0)
        pos = torch.tensor(res_pos_list, dtype=torch.float)

        return {
            'x': x,
            'pos': pos,
            'mol': mol
        }

    except Exception as e:
        logging.error(f"Error parsing receptor file {receptor_file}: {e}", exc_info=True)
        return None

def process_complex(pdb_id: str, data_dir: Path):
    """
    Main processing function for a single PDB ID.
    Finds the correct files and calls the respective parsers.
    
    Args:
        pdb_id (str): The 4-character PDB ID (e.g., '10gs').
        data_dir (Path): The root of the raw data (e.g., 'data/').
    
    Returns:
        dict or None: A nested dictionary {'ligand': {...}, 'receptor': {...}}
                      or None if any part fails.
    """
    complex_dir = data_dir / "v2020" / pdb_id
    ligand_file = complex_dir / f"{pdb_id}_ligand.sdf"
    receptor_file = complex_dir / f"{pdb_id}_protein.pdb"

    if not ligand_file.exists():
        logging.warning(f"Ligand file not found: {ligand_file}")
        return None
    if not receptor_file.exists():
        logging.warning(f"Receptor file not found: {receptor_file}")
        return None
        
    ligand_data = parse_ligand(ligand_file)
    if ligand_data is None:
        logging.warning(f"Failed to process ligand for {pdb_id}")
        return None
        
    receptor_data = parse_receptor(receptor_file)
    if receptor_data is None:
        logging.warning(f"Failed to process receptor for {pdb_id}")
        return None

    return {
        'ligand': ligand_data,
        'receptor': receptor_data
    }