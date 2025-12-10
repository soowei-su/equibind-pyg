# tests/test_geometry.py

import torch

from equibind_pyg.geometry.kabsch import (
    kabsch,
    apply_transform,
)
from equibind_pyg.geometry.metrics import (
    rmsd,
    centroid_distance,
    kabsch_rmsd,
)


def random_rotation_matrix(batch_size: int = 1) -> torch.Tensor:
    """
    Generate random proper rotation matrices R ∈ SO(3) using QR decomposition.

    Returns:
        R: [B, 3, 3]
    """
    Q = torch.randn(batch_size, 3, 3)
    Q, _ = torch.linalg.qr(Q)
    det = torch.det(Q)
    mask = det < 0
    if mask.any():
        Q[mask, :, -1] *= -1.0
    return Q


def test_kabsch_recovers_transform_single():
    """
    Check that Kabsch recovers a known SE(3) transform in the single-example case.
    """
    torch.manual_seed(0)
    N = 10

    # Ground truth points
    P = torch.randn(N, 3)  # [N, 3]

    # Random rotation + translation
    R_true = random_rotation_matrix(1).squeeze(0)  # [3, 3]
    t_true = torch.randn(3)                        # [3]

    # Apply transform: src = P_transformed, tgt = P
    src = P @ R_true.transpose(-1, -2) + t_true    # [N, 3]
    tgt = P.clone()

    R_est, t_est = kabsch(src, tgt)
    aligned = apply_transform(src, R_est.squeeze(0), t_est.squeeze(0))

    # RMSD between aligned and target should be tiny
    err = rmsd(aligned, tgt)
    assert err.item() < 1e-4, f"Kabsch alignment RMSD too high: {err.item()}"

    # Rotation should be close to true rotation
    # Rotation should be orthogonal with det ~ 1
    R_est_sq = R_est.squeeze(0)
    RtR = R_est_sq.T @ R_est_sq
    eye = torch.eye(3)
    assert torch.allclose(RtR, eye, atol=1e-4)
    det = torch.det(R_est_sq)
    assert det > 0.0 and abs(det - 1.0) < 1e-4


def test_kabsch_recovers_transform_batched():
    """
    Check that Kabsch works in the batched case.
    """
    torch.manual_seed(1)
    B, N = 4, 12

    P = torch.randn(B, N, 3)                      # [B, N, 3]
    R_true = random_rotation_matrix(B)            # [B, 3, 3]
    t_true = torch.randn(B, 3)                    # [B, 3]

    # src = P_transformed, tgt = P
    src = P @ R_true.transpose(-1, -2) + t_true.unsqueeze(-2)  # [B, N, 3]
    tgt = P.clone()

    R_est, t_est = kabsch(src, tgt)
    aligned = apply_transform(src, R_est, t_est)                # [B, N, 3]

    err = rmsd(aligned, tgt)  # [B]
    assert err.max().item() < 1e-4, f"Max batched Kabsch RMSD too high: {err.max().item()}"


def test_kabsch_rmsd_zero_when_aligned():
    """
    If src == tgt, Kabsch RMSD should be ~0.
    """
    torch.manual_seed(2)
    P = torch.randn(15, 3)
    val = kabsch_rmsd(P, P)
    assert val.item() < 1e-5, f"Kabsch RMSD for identical sets should be ~0, got {val.item()}"


def test_rmsd_simple_case():
    """
    Simple RMSD sanity check with known distance.
    """
    P = torch.zeros(10, 3)
    Q = torch.ones(10, 3)
    # distance between (0,0,0) and (1,1,1) = sqrt(3)
    expected = (3.0 ** 0.5)
    val = rmsd(P, Q)
    assert torch.allclose(val, torch.tensor(expected), atol=1e-6), (
        f"Expected RMSD {expected}, got {val.item()}"
    )


def test_centroid_distance_simple_case():
    """
    Centroid distance between two translated point clouds should match the translation magnitude.
    """
    torch.manual_seed(3)
    P = torch.randn(20, 3)
    shift = torch.tensor([2.0, -1.0, 0.5])
    Q = P + shift

    dist = centroid_distance(P, Q)
    expected = torch.linalg.norm(shift)
    assert torch.allclose(dist, expected, atol=1e-6), (
        f"Expected centroid distance {expected}, got {dist.item()}"
    )
