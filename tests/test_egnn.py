# tests/test_egnn.py

import torch
from torch_geometric.data import Data

from equibind_pyg.layers.egnn import EGNNLayer


def random_rotation_matrix() -> torch.Tensor:
    """
    Generate a random proper rotation matrix R ∈ SO(3) using QR decomposition.
    """
    Q = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(Q)
    # Ensure det(Q) = +1
    if torch.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


def build_fully_connected_edge_index(num_nodes: int) -> torch.Tensor:
    """
    Build a fully-connected (no self-loop) edge_index for a single graph.
    """
    row, col = torch.meshgrid(
        torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
    )
    edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)  # [2, N^2]
    mask = edge_index[0] != edge_index[1]  # remove self-loops
    return edge_index[:, mask]             # [2, E]


def test_egnn_equivariance_single_graph():
    """
    Test SE(3)-equivariance of EGNNLayer on a single graph.

    If we rotate+translate the input coordinates and run the same layer,
    the new coordinates should be the rotated+translated version of the
    original output, and the features should be identical.
    """
    torch.manual_seed(0)

    N = 6
    in_dim = 16
    out_dim = 16

    x = torch.randn(N, in_dim)
    pos = torch.randn(N, 3)
    edge_index = build_fully_connected_edge_index(N)

    layer = EGNNLayer(in_dim=in_dim, out_dim=out_dim)

    # Forward on original graph
    x_out1, pos_out1 = layer(x, pos, edge_index)

    # Apply a random SE(3) transform to inputs
    R = random_rotation_matrix()     # [3, 3]
    t = torch.randn(1, 3)            # [1, 3]

    pos_rot = pos @ R.T + t          # [N, 3]

    x_out2, pos_out2 = layer(x, pos_rot, edge_index)

    # Features should be invariant to global SE(3)
    assert torch.allclose(x_out1, x_out2, atol=1e-5), (
        "Node features are not invariant under global SE(3) transform."
    )

    # Coordinates should be equivariant:
    # pos_out1 rotated+translated ≈ pos_out2
    pos_out1_rot = pos_out1 @ R.T + t
    assert torch.allclose(pos_out1_rot, pos_out2, atol=1e-5), (
        "Coordinates are not equivariant under global SE(3) transform."
    )
