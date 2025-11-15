# Specification: EquiBind-PyG V1 (Rigid)

This document defines the scope and data schema for the V1 implementation.

**V1 Scope:** V1 will only implement **rigid re-docking**. The model will predict the rigid-body transformation (R, t) of the ligand.
* **Stretch Goals:** Flexible torsion fitting and LAS projection are considered out of scope for V1.

---

## Phase 1: Data Schema (HeteroData)

This blueprint defines the structure of the `torch_geometric.data.HeteroData` object that will represent each protein-ligand complex. This schema is critical for ensuring correct batching and message passing.

### Node Types

#### 1. `data['receptor']`
* **Description:** Represents the protein receptor as a graph of its residues. This is a coarse-grained representation.
* **`data['receptor'].x` (Node Features):**
    * **Type:** `torch.Tensor`
    * **Shape:** `[num_residues, F_r]` (e.g., `[num_residues, 21]`)
    * **Content:** One-hot encoding of the residue type (20 standard amino acids + 1 'other' category).
* **`data['receptor'].pos` (Node Coordinates):**
    * **Type:** `torch.Tensor`
    * **Shape:** `[num_residues, 3]`
    * **Content:** 3D coordinates of the $\alpha$-Carbon (C-Alpha) for each residue. This defines the geometry of the receptor graph.

#### 2. `data['ligand']`
* **Description:** Represents the small molecule ligand as an atom-level graph.
* **`data['ligand'].x` (Node Features):**
    * **Type:** `torch.Tensor`
    * **Shape:** `[num_atoms, F_l]` (e.g., `[num_atoms, 28]`)
    * **Content:** A concatenation of atomic features (e.g., one-hot atomic number, formal charge, is_aromatic flag, one-hot hybridization type).
* **`data['ligand'].pos` (Node Coordinates):**
    * **Type:** `torch.Tensor`
    * **Shape:** `[num_atoms, 3]`
    * **Content:** Ground-truth 3D coordinates for each atom. This serves as the ultimate prediction target for the model.

### Edge Types

#### 1. `data['ligand', 'intra', 'ligand'].edge_index`
* **Description:** Represents the covalent bonds *within* the ligand.
* **Type:** `torch.Tensor`
* **Shape:** `[2, num_bonds]`
* **Source:** Extracted directly from RDKit bond information. Defines the molecular graph topology.

#### 2. `data['receptor', 'intra', 'receptor'].edge_index`
* **Description:** Represents proximity connections *within* the receptor graph, enabling message passing between residues.
* **Type:** `torch.Tensor`
* **Shape:** `[2, num_receptor_edges]`
* **Source:** Dynamically calculated using `torch_geometric.nn.knn_graph` (e.g., k=30) or `torch_geometric.nn.radius_graph` (e.g., radius=10Å) based on the C-Alpha `pos` tensor.

