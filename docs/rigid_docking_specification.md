```
docs/rigid_docking_specification.md
```

---

# 📄 **Rigid Docking Specification – EquiBind-PyG (V1)**

This document defines the data structures, modeling assumptions, and system boundaries for **EquiBind-PyG V1**, a PyTorch Geometric implementation of the *rigid* component of EquiBind.
It reflects the capabilities implemented in this repository and clarifies design decisions relative to the original EquiBind codebase.

---

## **1. Scope of V1**

EquiBind-PyG V1 implements **rigid-body ligand docking** only.

### Included

* SE(3)-equivariant graph neural network (EGNN/IEGNN backbone)
* Rigid ligand pose prediction (rotation + translation applied to ligand atoms)
* Keypoint-based cross-attention interface between ligand and receptor graphs
* Pure PyG HeteroData pipelines for:

  * Ligand atom graph
  * Receptor residue graph
  * Intra-molecular and inter-molecular edges

### Excluded (Out of Scope)

These components from the original EquiBind paper are *not* implemented in V1:

* Flexible ligand torsion angle optimization
* Learned alignment for flexible docking
* OT-based pocket keypoint alignment
* Intersection / geometry regularization losses
* Full ligand evolution dynamics
* LAS (Local Atomic Structure) generation

The goal of V1 is to provide a **minimal, modular, research-grade PyG implementation** suitable as a template for downstream work and future extensions.

---

## **2. HeteroData Schema**

Each protein–ligand complex is represented as a `torch_geometric.data.HeteroData` object with two node types:

### **2.1 Receptor Node Type: `data['receptor']`**

| Attribute | Type          | Shape                | Description                                          |
| --------- | ------------- | -------------------- | ---------------------------------------------------- |
| `x`       | `FloatTensor` | `[num_residues, 21]` | 21-dim residue one-hot encoding (20 AAs + "unknown") |
| `pos`     | `FloatTensor` | `[num_residues, 3]`  | Cα coordinates used as the receptor graph geometry   |

Edge construction:

* `edge_index` generated via **k-NN** (default `k=30`)
* Captures spatial proximity between residues

This coarse-grained protein representation mirrors the original paper’s use of residue-level graphs and is computationally efficient.

---

### **2.2 Ligand Node Type: `data['ligand']`**

| Attribute | Type          | Shape             | Description                                     |
| --------- | ------------- | ----------------- | ----------------------------------------------- |
| `x`       | `FloatTensor` | `[num_atoms, 28]` | Atom-level features extracted from RDKit        |
| `pos`     | `FloatTensor` | `[num_atoms, 3]`  | Ground-truth ligand coordinates for supervision |

Ligand features (`F_l = 28`) include:

* One-hot atomic number
* Formal charge
* Aromaticity flag
* Hybridization type
* Donor/acceptor flags
  *(These features are a PyG adaptation of the original paper’s atom features; the paper does not prescribe a fixed feature set.)*

Edge construction:

* `edge_index` from RDKit bond graph

This preserves molecular connectivity exactly as defined by the chemical structure.

---

## **3. Edge Types**

### **3.1 Ligand Intra-Molecular Edges**

```
('ligand', 'intra', 'ligand')
```

* Covalent bonds obtained from RDKit
* Enable message passing within the ligand molecular graph

### **3.2 Receptor Intra-Molecular Edges**

```
('receptor', 'intra', 'receptor')
```

* Constructed using k-NN on Cα coordinates
* Captures geometric neighborhoods and local residue environment

No **ligand–receptor edges** are explicitly included in V1, consistent with the original EquiBind architecture, which relies on learned **keypoint cross-attention** rather than explicit cross-graph messaging.

---

## **4. Targets and Supervision Signals**

The model predicts:

* A set of **ligand positional updates**, producing final predicted coordinates:

  ```
  ligand_pos_pred: [N_ligand, 3]
  ```

The training objective computes:

* RMSD loss (main driver)
* Kabsch-aligned RMSD loss
* Centroid displacement loss

Each of these yields geometric supervision directly comparable to metrics reported in the original paper.

---

## **5. Loss Functions Included in V1**

The following losses are implemented and exposed in `equibind_pyg/models/losses.py`:

| Loss                          | Purpose                                         | Included?            |
| ----------------------------- | ----------------------------------------------- | -------------------- |
| **RMSD loss**                 | Fit raw coordinates                             | ✔ Yes                |
| **Kabsch RMSD**               | Penalize pose differences up to SE(3) alignment | ✔ Yes                |
| **Centroid loss**             | Encourage correct global placement              | ✔ Yes                |
| **OT keypoint loss**          | Pocket alignment                                | ✘ No (paper feature) |
| **Intersection loss**         | Penalize interpenetration                       | ✘ No                 |
| **Revised intersection loss** | Newer variant                                   | ✘ No                 |
| **Geometry regularization**   | Flexibility-related                             | ✘ No                 |
| **Torsion loss**              | Flexible ligand rotation                        | ✘ No                 |

These simplifications align with the rigid-only scope of V1.

---

## **6. Training and Evaluation**

### Training script (`scripts/train_rigid.py`)

* Loads `HeteroData` from disk
* Applies pre-processing if needed
* Builds the EquiBind-Rigid model
* Trains for *rigid pose prediction*
* Saves:

  * best-loss checkpoint
  * optional per-epoch checkpoints (commented-out)

### Evaluation script (`scripts/eval_rigid.py`)

* Loads trained checkpoint (supports strict or non-strict loading)
* Computes:

  * RMSD
  * Kabsch RMSD
  * Centroid distance
* Generates per-complex scatter plots (ground-truth vs predicted)

---

## **7. Relationship to the Original EquiBind Paper**

The original EquiBind architecture contains **two major modules**:

1. **Rigid alignment module** (keypoint-based attention + SE(3) equivariant updates)
2. **Flexible refinement module** (torsion prediction + geometry regularization)

V1 implements Module (1) faithfully using PyG and omits Module (2) entirely.

The original paper does *not* prescribe:

* A fixed residue feature dimensionality
* A fixed atom feature specification
* A PyG-compatible batching strategy

These components are engineering design decisions made for the purposes of modularity, reproducibility, and PyG compatibility.

Thus, V1 is faithful to the paper’s **architecture**, but more structured from a **software engineering** perspective.

---

## **8. Future Extensions (V2 and Beyond)**

V1 intentionally isolates rigid docking. Future work may include:

### **Flexible docking extensions**

* Torsion-angle regressors
* Ligand evolution layers
* Full geometry regularization losses

### **Cross-graph enhancements**

* Learned ligand–receptor edges
* Pocket extraction modules
* OT-based cross-attention

### **Improved PyG integration**

* Registering dataset with PyG’s dataset registry
* Exposing models under `torch_geometric.nn.equibind`
* Adding unit tests to PyG’s CI suite

---

## **9. Summary**

This specification defines a clean, PyG-focused, V1 implementation of rigid EquiBind.
It provides:

* Reproducible data structures
* A faithful SE(3)-equivariant architecture
* A simplified, research-ready loss suite
* A modular codebase designed for upstream PR review


