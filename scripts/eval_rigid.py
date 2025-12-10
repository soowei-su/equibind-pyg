"""
Evaluation script for the EquiBind-Rigid PyG model.

This script:

- Loads configuration from a YAML file.
- Builds the PDBBindPyG dataset for a given ID list.
- Instantiates the EquiBindRigid model.
- Loads a trained checkpoint.
- Evaluates average:
    - total loss
    - RMSD
    - Kabsch RMSD
    - centroid distance
- Optionally saves a small number of pose visualizations.

Example usage:

    python -m scripts.eval_rigid \
        --config configs/rigid_default.yaml \
        --ids data/test_ids.txt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
import torch
from torch_geometric.loader import DataLoader

from equibind_pyg.data.pdbbind_dataset import PDBBindPyG
from equibind_pyg.models.equibind_rigid import EquiBindRigid
from equibind_pyg.models.losses import compute_loss

import os
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def scatter_pair(pos_gt, pos_pred, title: str, out_dir: str = "figures/poses_eval") -> None:
    """
    Save a 3D scatter plot comparing ground truth and predicted ligand positions.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], s=10, label="GT")
    ax.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], s=10, label="Pred")
    ax.set_title(title)
    ax.legend()
    fig.savefig(os.path.join(out_dir, f"{title}.png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rigid_default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        required=True,
        help="Path to IDs file (one PDB ID per line).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config if set).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    tr_cfg = cfg.get("training", {})
    ckpt_cfg = cfg["checkpoint"]

    device = torch.device(
        tr_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    # ------------------------------------------------------------------ #
    # Dataset & DataLoader
    # ------------------------------------------------------------------ #
    logger.info(
        "Loading dataset from %s with IDs from %s...",
        ds_cfg["root"],
        args.ids,
    )
    ds = PDBBindPyG(
        root=ds_cfg["root"],
        id_file=args.ids,
        k_receptor_neighbors=ds_cfg["k_receptor_neighbors"],
        force_reprocess=False,
    )
    if len(ds) == 0:
        logger.error("Dataset is empty. Did you run preprocessing?")
        return

    batch_size = args.batch if args.batch is not None else ds_cfg["batch_size"]
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # ------------------------------------------------------------------ #
    # Model & checkpoint
    # ------------------------------------------------------------------ #
    model = EquiBindRigid(
        node_dim=model_cfg["node_dim"],
        num_layers=model_cfg["num_layers"],
        num_keypoints=model_cfg["num_keypoints"],
        edge_mlp_dim=model_cfg["edge_mlp_dim"],
        coord_mlp_dim=model_cfg["coord_mlp_dim"],
        cross_attn_dim=model_cfg["cross_attn_dim"],
        lig_in_dim=28,
        rec_in_dim=21,
    ).to(device)


    ckpt_path = Path(ckpt_cfg["save_path"])
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return

    logger.info("Loading checkpoint from %s", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state_dict, strict=True)
    logger.info("Checkpoint loaded successfully.")

    model.eval()

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #
    total_loss = 0.0
    total_rmsd = 0.0
    total_kabsch = 0.0
    total_centroid = 0.0
    count = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data)

            # Compute loss dictionary
            loss_dict = compute_loss(
                out,
                data,
                lambda_rmsd=cfg["loss"]["lambda_rmsd"],
                lambda_kabsch=cfg["loss"]["lambda_kabsch"],
                lambda_centroid=cfg["loss"]["lambda_centroid"],
            )
            loss = loss_dict["total"]

            total_loss += float(loss.item())
            total_rmsd += float(loss_dict["rmsd"].item())
            total_kabsch += float(loss_dict["kabsch_rmsd"].item())
            total_centroid += float(loss_dict["centroid"].item())
            count += 1

            # Optional pose visualizations
            pred = out["ligand_pos_pred"].detach().cpu().numpy()
            target = data.ligand_pos_bound.detach().cpu().numpy()
            pdb_id = getattr(data, "pdb_id", f"batch_{count}")
            title = f"pose_{pdb_id}"
            scatter_pair(target, pred, title)

    if count == 0:
        logger.warning("No batches processed during evaluation.")
        return

    avg_loss = total_loss / count
    avg_rmsd = total_rmsd / count
    avg_kabsch = total_kabsch / count
    avg_centroid = total_centroid / count

    logger.info("Evaluation over %d batches:", count)
    logger.info("  Loss (avg)         : %.3f", avg_loss)
    logger.info("  RMSD (avg)         : %.3f Å", avg_rmsd)
    logger.info("  Kabsch RMSD (avg)  : %.5f Å", avg_kabsch)
    logger.info("  Centroid Dist (avg): %.3f Å", avg_centroid)


if __name__ == "__main__":
    main()
