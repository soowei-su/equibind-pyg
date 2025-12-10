# EquiBind-PyG (Rigid)

A PyTorch Geometric (PyG) reimplementation of the rigid-body component of **EquiBind**, the SE(3)-equivariant neural network for protein–ligand docking.

This repository focuses on:

* A clean, modular PyG version of **rigid EquiBind** (rotation + translation only).
* A simplified but stable **geometric loss** (RMSD + Kabsch + centroid).
* A PyG-compliant **HeteroData PDBBind dataset**.
* An SE(3)-equivariant **Interaction-EGNN** architecture.

It is designed for:

* CS224W course project requirements,
* Reproducible research,
* Future PyG integration.

For full feature definitions (28-dim ligand, 21-dim residue), see `docs/rigid_docking_specification.md`.

---

## 1. Installation

Clone and create environment:

```bash
git clone https://github.com/yourname/equibind-pyg.git
cd equibind-pyg

conda env create -f environment.yml
conda activate equibind-pyg

pip install -e .
```

---

## 2. Dataset — PDBBind v2020

Download and extract PDBBind General Set (2020) under:

```
data/v2020/<PDB_ID>/
    <id>_ligand.sdf
    <id>_protein.pdb
```

### Dataset Splits

Provide:

```
data/train_ids.txt
data/val_ids.txt
data/test_ids.txt
```

One PDB ID per line.

---

## 3. Automatic Data Processing

Running the dataset triggers:

* ligand parsing (`.sdf`)
* receptor atom graph construction (`.pdb`)
* ligand (28-dim) + receptor (21-dim) feature creation
* edge construction (ligand–ligand, receptor–receptor, ligand–receptor)
* caching to:

```
data/pyg_data/processed/*.pt
```

No separate preprocessing script required.

---

## 4. Training

Edit model & training configuration:

```
configs/rigid_default.yaml
```

Run:

```bash
python -m scripts.train_rigid \
  --config configs/rigid_default.yaml \
  --ids data/train_ids.txt
```

Outputs:

* CSV metrics log: `logs/train_metrics.csv`
* RMSD/centroid plots:

  * `fig_rmsd_curve.png`
  * `fig_centroid_curve.png`
* Best model checkpoint:

  * `checkpoints/equibind_rigid_default.pt`
* Optional per-epoch checkpoints (commented in code)

Logging format:

```
[Epoch 031] Loss = 39.590 | RMSD = 38.731 Å | Kabsch = 0.00050 Å | Centroid = 17.164 Å
```

---

## 5. Evaluation

Evaluate trained model:

```bash
python -m scripts.eval_rigid \
  --config configs/rigid_default.yaml \
  --ids data/test_ids.txt
```

Typical output:

```
Loss (avg)         : 44.823
RMSD (avg)         : 43.136 Å
Kabsch RMSD (avg)  : 0.00010 Å
Centroid Dist (avg): 16.872 Å
```

Qualitative pose visualizations saved to:

```
figures/poses_eval/
```

---

## 6. Overfitting Sanity Check

Create tiny subset:

```bash
shuf data/train_ids.txt | head -n 15 > data/overfit_ids.txt
```

Train:

```bash
python -m scripts.train_rigid \
  --config configs/overfit_small.yaml \
  --ids data/overfit_ids.txt
```

Evaluate:

```bash
python -m scripts.eval_rigid \
  --config configs/overfit_small.yaml \
  --ids data/overfit_ids.txt
```

Expected: RMSD drops substantially → confirms correct gradients and geometry.

---

## 7. Repository Structure

```
equibind_pyg/
  data/
    pdbbind_dataset.py       # PDBBind → PyG dataset + caching
    preprocess.py            # Ligand/receptor parsing & featurization
  layers/
    egnn.py                  # Base EGNN layer (SE(3)-equivariant)
    iegnn.py                 # Interaction-EGNN for cross-attention
    keypoint_attention.py    # (included for completeness, unused in rigid baseline)
  geometry/
    kabsch.py                # Differentiable Kabsch alignment
    metrics.py               # RMSD, Kabsch RMSD, centroid distance
  models/
    equibind_rigid.py        # Rigid EquiBind forward pass
    losses.py                # RMSD + Kabsch + centroid loss

scripts/
  train_rigid.py             # Training loop
  eval_rigid.py              # Evaluation & pose Viz

configs/
  rigid_default.yaml         # Main model config
  overfit_small.yaml         # Small overfitting config

docs/
  v1_rigid_spec.md           # Full feature schema

figures/
  fig_rmsd_curve.png
  fig_centroid_curve.png
  poses_eval/

logs/
  train_metrics.csv

checkpoints/
  equibind_rigid_default.pt
  rigid_overfit.pt
```

---

## 8. Limitations

This implementation does **not** include several components from the full EquiBind model:

* Optimal transport loss (OT)
* Geometric intersection loss
* Learned torsion-angle updates (flexible docking)
* LAS collision regularizers

Thus, results are **not directly comparable** to the original paper.
This project is a **clean PyG baseline for rigid docking**.

---

## 9. Acknowledgements

Original research:
**EquiBind: Geometric Deep Learning for Protein–Ligand Docking**
Hannes Stärk et al., ICML 2022
[https://github.com/HannesStark/EquiBind](https://github.com/HannesStark/EquiBind)

Developed for:
**Stanford CS224W — Machine Learning with Graphs (2025)**
Course Project: *EquiBind-PyG: A Modular Equivariant GNN for Rigid Protein–Ligand Docking*



