# tests/test_loss.py

import torch

from equibind_pyg.models.losses import compute_loss
from equibind_pyg.geometry.metrics import rmsd, kabsch_rmsd, centroid_distance


class DummyData:
    """
    Minimal stand-in for a HeteroData object, with only
    the attribute needed by compute_loss: ligand_pos_bound.
    """
    def __init__(self, ligand_pos_bound: torch.Tensor):
        self.ligand_pos_bound = ligand_pos_bound


def test_compute_loss_zero_when_perfect_prediction():
    """
    If predicted ligand positions exactly match ground truth,
    all metrics (rmsd, kabsch_rmsd, centroid) should be ~0
    and total loss should be ~0.
    """
    torch.manual_seed(0)
    N = 10

    target = torch.randn(N, 3)
    pred = target.clone()

    out = {"ligand_pos_pred": pred}
    data = DummyData(ligand_pos_bound=target)

    loss_dict = compute_loss(out, data)

    assert loss_dict["rmsd"].item() < 1e-6
    assert loss_dict["kabsch_rmsd"].item() < 1e-5
    assert loss_dict["centroid"].item() < 1e-6
    # total = λ_rmsd * rmsd + λ_kabsch * kabsch_rmsd + λ_centroid * centroid
    assert loss_dict["total"].item() < 1e-5


def test_compute_loss_scaling_with_lambdas():
    """
    Check that changing the lambda weights in compute_loss changes
    the total loss in the expected linear way.
    """
    torch.manual_seed(1)
    N = 8

    target = torch.zeros(N, 3)
    pred = torch.ones(N, 3)  # distance sqrt(3) per atom

    out = {"ligand_pos_pred": pred}
    data = DummyData(ligand_pos_bound=target)

    # First, compute individual metrics directly
    r_val = rmsd(pred, target)
    k_val = kabsch_rmsd(pred, target)
    c_val = centroid_distance(pred, target)

    # Case 1: all lambdas = 1
    loss1 = compute_loss(
        out,
        data,
        lambda_rmsd=1.0,
        lambda_kabsch=1.0,
        lambda_centroid=1.0,
    )

    expected1 = r_val + k_val + c_val
    assert torch.allclose(loss1["total"], expected1, atol=1e-6)

    # Case 2: scale RMSD by 2, others by 0
    loss2 = compute_loss(
        out,
        data,
        lambda_rmsd=2.0,
        lambda_kabsch=0.0,
        lambda_centroid=0.0,
    )
    expected2 = 2.0 * r_val
    assert torch.allclose(loss2["total"], expected2, atol=1e-6)
