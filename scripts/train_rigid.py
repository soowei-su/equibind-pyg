"""
Training script for the EquiBind-Rigid PyG model.

This script:

- Loads configuration from a YAML file.
- Builds the PDBBindPyG dataset for a given ID list.
- Instantiates the EquiBindRigid model.
- Trains for a specified number of epochs, logging:
  - total loss
  - RMSD
  - Kabsch RMSD
  - centroid distance
- Writes a CSV of per-epoch metrics under logs/train_metrics.csv.
- Saves the best-loss checkpoint to the path specified in the config.

Example usage:

    python -m scripts.train_rigid \
        --config configs/rigid_default.yaml \
        --ids data/train_ids.txt
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

import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    tr_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
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

    loader = DataLoader(
        ds,
        batch_size=ds_cfg["batch_size"],
        shuffle=True,
    )

    # ------------------------------------------------------------------ #
    # Model & Optimizer
    # ------------------------------------------------------------------ #
    model = EquiBindRigid(
        node_dim=model_cfg["node_dim"],
        num_layers=model_cfg["num_layers"],
        num_keypoints=model_cfg["num_keypoints"],
        edge_mlp_dim=model_cfg["edge_mlp_dim"],
        coord_mlp_dim=model_cfg["coord_mlp_dim"],
        cross_attn_dim=model_cfg["cross_attn_dim"],
        lig_in_dim=28,   # ligand feature dim from preprocess
        rec_in_dim=21,   # receptor feature dim from preprocess
    ).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
    )

    # ------------------------------------------------------------------ #
    # Metrics CSV & best checkpoint tracking
    # ------------------------------------------------------------------ #
    log_path = Path("logs/train_metrics.csv")
    log_path.parent.mkdir(exist_ok=True)

    # Start fresh each run
    with open(log_path, "w") as f:
        f.write("epoch,loss,rmsd,kabsch,centroid\n")

    best_loss = float("inf")
    best_epoch = -1
    best_ckpt_path = Path(ckpt_cfg["save_path"])
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    logger.info("Starting training for %d epochs...", tr_cfg["epochs"])

    num_epochs = tr_cfg["epochs"]
    grad_clip_norm = tr_cfg.get("grad_clip_norm", 0.0) or 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_loss = 0.0
        total_rmsd = 0.0
        total_kabsch = 0.0
        total_centroid = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(loader):
            data = data.to(device)

            out = model(data)

            loss_dict = compute_loss(
                out,
                data,
                lambda_rmsd=loss_cfg["lambda_rmsd"],
                lambda_kabsch=loss_cfg["lambda_kabsch"],
                lambda_centroid=loss_cfg["lambda_centroid"],
            )
            loss = loss_dict["total"]

            # Skip NaN/Inf batches to keep training stable
            if not torch.isfinite(loss):
                logger.warning(
                    "Encountered NaN/Inf loss at batch %d; skipping. "
                    "RMSD=%.3e, Kabsch=%.3e, Centroid=%.3e",
                    batch_idx,
                    loss_dict["rmsd"].item(),
                    loss_dict["kabsch_rmsd"].item(),
                    loss_dict["centroid"].item(),
                )
                continue

            optimizer.zero_grad()
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

            total_loss += float(loss.detach())
            total_rmsd += float(loss_dict["rmsd"].detach())
            total_kabsch += float(loss_dict["kabsch_rmsd"].detach())
            total_centroid += float(loss_dict["centroid"].detach())
            num_batches += 1

        if num_batches == 0:
            logger.warning(
                "Epoch %d: no valid batches (all NaN/Inf); skipping logging.",
                epoch,
            )
            continue

        avg_loss = total_loss / num_batches
        avg_rmsd = total_rmsd / num_batches
        avg_kabsch = total_kabsch / num_batches
        avg_centroid = total_centroid / num_batches

        # Your requested log format:
        logger.info(
            "[Epoch %03d] Loss = %.3f | RMSD = %.3f Å | Kabsch = %.5f Å | Centroid = %.3f Å",
            epoch,
            avg_loss,
            avg_rmsd,
            avg_kabsch,
            avg_centroid,
        )

        # Append this epoch’s metrics to CSV
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss},{avg_rmsd},{avg_kabsch},{avg_centroid}\n")

        # Track best checkpoint by lowest total loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(
                "New best model at epoch %03d with loss %.3f saved to %s",
                best_epoch,
                best_loss,
                best_ckpt_path,
            )

        # Optionally save per-epoch checkpoints (kept commented for research use)
        # epoch_ckpt_path = Path(f"checkpoints/equibind_rigid_epoch_{epoch:03d}.pt")
        # epoch_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        # torch.save(model.state_dict(), epoch_ckpt_path)
        # logger.info("Saved epoch checkpoint: %s", epoch_ckpt_path)

    logger.info(
        "Training finished. Best epoch: %03d with loss %.3f (checkpoint: %s)",
        best_epoch,
        best_loss,
        best_ckpt_path,
    )

    # ------------------------------------------------------------------ #
    # Plot training curves (RMSD & centroid)
    # ------------------------------------------------------------------ #
    df = pd.read_csv(log_path)

    plt.figure()
    plt.plot(df["epoch"], df["rmsd"])
    plt.xlabel("Epoch")
    plt.ylabel("RMSD (Å)")
    plt.title("Training RMSD over epochs")
    plt.grid(True)
    plt.savefig("figures/fig_rmsd_curve.png")

    plt.figure()
    plt.plot(df["epoch"], df["centroid"])
    plt.xlabel("Epoch")
    plt.ylabel("Centroid distance (Å)")
    plt.title("Training centroid distance over epochs")
    plt.grid(True)
    plt.savefig("figures/fig_centroid_curve.png")


if __name__ == "__main__":
    main()
