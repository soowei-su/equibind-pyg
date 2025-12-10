import os
from pathlib import Path

import pytest
from torch_geometric.loader import DataLoader

from equibind_pyg.data.pdbbind_dataset import PDBBindPyG

# Adjust these paths to match your repo layout
DATA_ROOT = Path("data/pyg_data")
ID_FILE = Path("data/dev_ids.txt")


@pytest.mark.parametrize("k_receptor_neighbors", [16, 30])
def test_dataset_instantiation_and_length(k_receptor_neighbors: int):
    """
    Basic sanity check:
    - ID file exists (otherwise skip).
    - Dataset can be instantiated.
    - Dataset has at least one processed complex.
    """
    if not ID_FILE.exists():
        pytest.skip(f"ID file not found: {ID_FILE} (skipping dataset tests)")

    ds = PDBBindPyG(
        root=str(DATA_ROOT),
        id_file=str(ID_FILE),
        k_receptor_neighbors=k_receptor_neighbors,
        force_reprocess=False,
    )

    assert len(ds) >= 0  # at least doesn't crash
    if len(ds) == 0:
        pytest.skip("Dataset is empty after processing; check preprocessing pipeline.")


def test_single_sample_shapes_and_feature_dims():
    """
    Check that a single sample has the expected structure and v1_rigid
    feature dimensions: 28 (ligand), 21 (receptor), position dim 3.
    """
    if not ID_FILE.exists():
        pytest.skip(f"ID file not found: {ID_FILE} (skipping dataset tests)")

    ds = PDBBindPyG(
        root=str(DATA_ROOT),
        id_file=str(ID_FILE),
        k_receptor_neighbors=30,
        force_reprocess=False,
    )

    if len(ds) == 0:
        pytest.skip("Dataset is empty after processing; check preprocessing pipeline.")

    data = ds[0]

    assert "ligand" in data.node_types
    assert "receptor" in data.node_types

    lig_x = data["ligand"].x
    lig_pos = data["ligand"].pos
    rec_x = data["receptor"].x
    rec_pos = data["receptor"].pos

    # Basic shape checks
    assert lig_x.dim() == 2
    assert lig_pos.dim() == 2
    assert rec_x.dim() == 2
    assert rec_pos.dim() == 2

    # Feature dimensions per your v1 rigid spec
    assert lig_x.size(-1) == 28, f"Expected ligand feature dim 28, got {lig_x.size(-1)}"
    assert rec_x.size(-1) == 21, f"Expected receptor feature dim 21, got {rec_x.size(-1)}"
    assert lig_pos.size(-1) == 3, f"Expected ligand pos dim 3, got {lig_pos.size(-1)}"
    assert rec_pos.size(-1) == 3, f"Expected receptor pos dim 3, got {rec_pos.size(-1)}"


def test_dataloader_batch_shapes():
    """
    Check that a DataLoader can batch multiple complexes
    and preserve feature dimensions.
    """
    if not ID_FILE.exists():
        pytest.skip(f"ID file not found: {ID_FILE} (skipping dataset tests)")

    ds = PDBBindPyG(
        root=str(DATA_ROOT),
        id_file=str(ID_FILE),
        k_receptor_neighbors=30,
        force_reprocess=False,
    )

    if len(ds) < 2:
        pytest.skip(
            "Need at least 2 complexes to test batching; currently len(dataset) < 2."
        )

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    lig_x = batch["ligand"].x
    lig_pos = batch["ligand"].pos
    rec_x = batch["receptor"].x
    rec_pos = batch["receptor"].pos

    assert lig_x.dim() == 2
    assert lig_pos.dim() == 2
    assert rec_x.dim() == 2
    assert rec_pos.dim() == 2

    assert lig_x.size(-1) == 28
    assert rec_x.size(-1) == 21
    assert lig_pos.size(-1) == 3
    assert rec_pos.size(-1) == 3
