V1 will only implement rigid re-docking. Torsion fitting and LAS projection are stretch goals.

## Phase 1: Data Schema (HeteroData)

This blueprint defines the structure of the `torch_geometric.data.HeteroData` object that will represent each protein-ligand complex.

### Node Types

#### 1. `data['receptor']`
* **Description:** Represents the protein receptor as a graph of its residues.
* **`data['receptor'].x` (Node Features):**
    * Type: `torch.Tensor`
    * Shape: `[num_residues, F_r]`
    * Content: One-hot encoding of the residue type (e.g., ALA, CYS, ... 20 standard types).
* **`data['receptor'].pos` (Node Coordinates):**
    * Type: `torch.Tensor`
    * Shape: `[num_residues, 3]`
    * Content: 3D coordinates of the $\alpha$-Carbon (C-Alpha) for each residue.

#### 2. `data['ligand']`
* **Description:** Represents the small molecule ligand as a graph of its atoms.
* **`data['ligand'].x` (Node Features):**
    * Type: `torch.Tensor`
    * Shape: `[num_atoms, F_l]`
    * Content: A concatenation of atomic features (e.g., atomic number, formal charge, is_aromatic flag, hybridization).
* **`data['ligand'].pos` (Node Coordinates):**
    * Type: `torch.Tensor`
    * Shape: `[num_atoms, 3]`
    * Content: Ground-truth 3D coordinates for each atom. This will also serve as the prediction target.

### Edge Types

#### 1. `data['ligand', 'intra', 'ligand'].edge_index`
* **Description:** Represents the covalent bonds *within* the ligand.
* **Type:** `torch.Tensor`
* **Shape:** `[2, num_bonds]`
* **Source:** Extracted directly from RDKit.

#### 2. `data['receptor', 'intra', 'receptor'].edge_index`
* **Description:** Represents proximity connections *within* the receptor.
* **Type:** `torch.Tensor`
* **Shape:** `[2, num_receptor_edges]`
* **Source:** Will be dynamically calculated using `torch_geometric.nn.knn_graph` (e.g., k=30) or `torch_geometric.nn.radius_graph` (e.g., radius=10Å) based on C-Alpha positions.

