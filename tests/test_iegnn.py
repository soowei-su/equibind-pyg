# tests/test_iegnn.py

import torch

from equibind_pyg.layers.iegnn import IEGNNLayer


def random_rotation_matrix() -> torch.Tensor:
    """
    Generate a random proper rotation matrix R ∈ SO(3) using QR decomposition.
    """
    Q = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(Q)
    if torch.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


def fully_connected_edge_index(num_nodes: int) -> torch.Tensor:
    """
    Build a fully connected (no self-loops) edge_index for a single graph.
    """
    row, col = torch.meshgrid(
        torch.arange(num_nodes),
        torch.arange(num_nodes),
        indexing="ij",
    )
    edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)  # [2, N^2]
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]  # [2, E]


def test_iegnn_shapes_and_forward():
    """
    Basic sanity: IEGNNLayer runs and returns correct shapes for
    ligand + receptor.
    """
    torch.manual_seed(0)

    N_lig = 5
    N_rec = 7
    node_dim = 16

    lig_x = torch.randn(N_lig, node_dim)
    lig_pos = torch.randn(N_lig, 3)
    lig_edge_index = fully_connected_edge_index(N_lig)

    rec_x = torch.randn(N_rec, node_dim)
    rec_pos = torch.randn(N_rec, 3)
    rec_edge_index = fully_connected_edge_index(N_rec)

    layer = IEGNNLayer(node_dim=node_dim)

    lig_x_out, lig_pos_out, rec_x_out, rec_pos_out = layer(
        lig_x,
        lig_pos,
        lig_edge_index,
        rec_x,
        rec_pos,
        rec_edge_index,
    )

    assert lig_x_out.shape == (N_lig, node_dim)
    assert lig_pos_out.shape == (N_lig, 3)
    assert rec_x_out.shape == (N_rec, node_dim)
    assert rec_pos_out.shape == (N_rec, 3)


def test_iegnn_se3_equivariance():
    """
    Test that IEGNNLayer is SE(3)-equivariant in the expected way:

    - Node features are invariant to global rotations/translations.
    - Coordinates are equivariant:
          pos_out2 ≈ (pos_out1 @ R^T + t)
    """
    torch.manual_seed(1)

    N_lig = 6
    N_rec = 8
    node_dim = 16

    lig_x = torch.randn(N_lig, node_dim)
    lig_pos = torch.randn(N_lig, 3)
    lig_edge_index = fully_connected_edge_index(N_lig)

    rec_x = torch.randn(N_rec, node_dim)
    rec_pos = torch.randn(N_rec, 3)
    rec_edge_index = fully_connected_edge_index(N_rec)

    layer = IEGNNLayer(node_dim=node_dim)

    # Forward on original
    lig_x1, lig_pos1, rec_x1, rec_pos1 = layer(
        lig_x,
        lig_pos,
        lig_edge_index,
        rec_x,
        rec_pos,
        rec_edge_index,
    )

    # Apply a global SE(3) transform to BOTH ligand and receptor
    R = random_rotation_matrix()  # [3, 3]
    t = torch.randn(1, 3)         # [1, 3]

    lig_pos_rot = lig_pos @ R.T + t
    rec_pos_rot = rec_pos @ R.T + t

    lig_x2, lig_pos2, rec_x2, rec_pos2 = layer(
        lig_x,
        lig_pos_rot,
        lig_edge_index,
        rec_x,
        rec_pos_rot,
        rec_edge_index,
    )

    # Features should be invariant
    assert torch.allclose(lig_x1, lig_x2, atol=1e-5), (
        "Ligand features are not invariant under global SE(3) transform."
    )
    assert torch.allclose(rec_x1, rec_x2, atol=1e-5), (
        "Receptor features are not invariant under global SE(3) transform."
    )

    # Coordinates should be equivariant
    lig_pos1_rot = lig_pos1 @ R.T + t
    rec_pos1_rot = rec_pos1 @ R.T + t

    assert torch.allclose(lig_pos1_rot, lig_pos2, atol=1e-5), (
        "Ligand coordinates are not equivariant under global SE(3) transform."
    )
    assert torch.allclose(rec_pos1_rot, rec_pos2, atol=1e-5), (
        "Receptor coordinates are not equivariant under global SE(3) transform."
    )
