#!/usr/bin/env python
"""
Sanity checker for the PDBBindPyG dataset.

This script instantiates a PDBBindPyG dataset, inspects a few
individual complexes, and verifies that:

- ligand and receptor node feature dimensions are as expected
- positions have the correct shape
- batching via DataLoader works as intended

Example usage (from repo root):

    python -m scripts.check_dataset \
        --root data/pyg_data \
        --id-file data/dev_ids.txt \
        --batch-size 4 \
        --num-samples 3 \
        --force-reprocess 0
"""

import argparse
import logging
from pathlib import Path

from torch_geometric.loader import DataLoader

from equibind_pyg.data.pdbbind_dataset import PDBBindPyG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/pyg_data",
        help="Root directory where the PyG 'processed/' folder lives (e.g., data/pyg_data).",
    )
    parser.add_argument(
        "--id-file",
        type=str,
        default="data/dev_ids.txt",
        help="Path to text file containing PDB IDs, one per line.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoader sanity check.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="How many individual complexes to inspect.",
    )
    parser.add_argument(
        "--k-receptor-neighbors",
        type=int,
        default=30,
        help="k for k-NN graph on receptor residues.",
    )
    parser.add_argument(
        "--force-reprocess",
        type=int,
        default=0,
        help="If 1, delete all processed .pt files and re-process from scratch.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    id_file = Path(args.id_file)

    if not id_file.exists():
        logger.error("ID file not found: %s", id_file)
        return

    logger.info("Instantiating PDBBindPyG(root=%s, id_file=%s)", root, id_file)

    dataset = PDBBindPyG(
        root=str(root),
        id_file=str(id_file),
        k_receptor_neighbors=args.k_receptor_neighbors,
        force_reprocess=bool(args.force_reprocess),
    )

    logger.info("Dataset length (processed_ids): %d", len(dataset))
    if len(dataset) == 0:
        logger.warning("PDBBindPyG dataset is empty. Nothing to check.")
        return

    # ----------------------------------------------------------------------
    # Inspect a few individual samples
    # ----------------------------------------------------------------------
    num_samples = min(args.num_samples, len(dataset))
    logger.info("Inspecting %d individual complexes...", num_samples)

    for i in range(num_samples):
        data = dataset[i]
        pdb_id = getattr(data, "pdb_id", f"idx_{i}")
        logger.info("Sample %d / %d — pdb_id=%s", i + 1, num_samples, pdb_id)

        # Ligand
        lig_x = data["ligand"].x
        lig_pos = data["ligand"].pos
        lig_edge_index = data["ligand", "intra", "ligand"].edge_index

        # Receptor
        rec_x = data["receptor"].x
        rec_pos = data["receptor"].pos
        rec_edge_index = data["receptor", "intra", "receptor"].edge_index

        logger.info("  ligand.x shape:      %s", tuple(lig_x.shape))
        logger.info("  ligand.pos shape:    %s", tuple(lig_pos.shape))
        logger.info(
            "  ligand.edge_index:   %s (E=%d)",
            tuple(lig_edge_index.shape),
            lig_edge_index.size(1),
        )

        logger.info("  receptor.x shape:    %s", tuple(rec_x.shape))
        logger.info("  receptor.pos shape:  %s", tuple(rec_pos.shape))
        logger.info(
            "  receptor.edge_index: %s (E=%d)",
            tuple(rec_edge_index.shape),
            rec_edge_index.size(1),
        )

        # Validate against rigid v1 feature expectations
        assert lig_x.size(-1) == 28, f"Expected ligand feature dim 28, got {lig_x.size(-1)}"
        assert rec_x.size(-1) == 21, f"Expected receptor feature dim 21, got {rec_x.size(-1)}"
        assert lig_pos.size(-1) == 3, f"Expected ligand pos dim 3, got {lig_pos.size(-1)}"
        assert rec_pos.size(-1) == 3, f"Expected receptor pos dim 3, got {rec_pos.size(-1)}"

    # ----------------------------------------------------------------------
    # Check a DataLoader batch
    # ----------------------------------------------------------------------
    logger.info("Creating DataLoader with batch_size=%d", args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    batch = next(iter(loader))
    logger.info("Batched sample:")
    logger.info("  ligand.x batch shape:     %s", tuple(batch["ligand"].x.shape))
    logger.info("  ligand.pos batch shape:   %s", tuple(batch["ligand"].pos.shape))
    logger.info("  receptor.x batch shape:   %s", tuple(batch["receptor"].x.shape))
    logger.info("  receptor.pos batch shape: %s", tuple(batch["receptor"].pos.shape))

    logger.info("Dataset + DataLoader checks passed.")


if __name__ == "__main__":
    main()

