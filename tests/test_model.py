# tests/test_model.py

import torch
from torch_geometric.data import HeteroData

from equibind_pyg.models.equibind_rigid import EquiBindRigid


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


def build_toy_heterodata(
    n_lig: int = 6,
    n_rec: int = 10,
    node_dim: int = 32,
) -> HeteroData:
    """
    Build a toy single-complex HeteroData with random features and positions.
    """
    torch.manual_seed(0)

    data = HeteroData()

    # Ligand
    lig_x = torch.randn(n_lig, node_dim)
    lig_pos = torch.randn(n_lig, 3)
    lig_edge_index = fully_connected_edge_index(n_lig)

    data["ligand"].x = lig_x
    data["ligand"].pos = lig_pos
    data["ligand", "intra", "ligand"].edge_index = lig_edge_index

    # Receptor
    rec_x = torch.randn(n_rec, node_dim)
    rec_pos = torch.randn(n_rec, 3)
    rec_edge_index = fully_connected_edge_index(n_rec)

    data["receptor"].x = rec_x
    data["receptor"].pos = rec_pos
    data["receptor", "intra", "receptor"].edge_index = rec_edge_index

    return data


def test_equibind_rigid_forward_shapes():
    """
    Basic forward-pass sanity check:
    - Model runs without error.
    - Output dictionary has expected keys and shapes.
    """
    torch.manual_seed(1)

    node_dim = 32
    num_layers = 2
    num_keypoints = 8

    data = build_toy_heterodata(n_lig=5, n_rec=9, node_dim=node_dim)

    model = EquiBindRigid(
        node_dim=node_dim,
        num_layers=num_layers,
        num_keypoints=num_keypoints,
        edge_mlp_dim=32,
        coord_mlp_dim=32,
        cross_attn_dim=32,
    )

    out = model(data)

    assert "ligand_pos_pred" in out
    assert "ligand_x" in out
    assert "receptor_x" in out
    assert "ligand_kp_pos" in out
    assert "receptor_kp_pos" in out
    assert "R" in out
    assert "t" in out

    N_lig = data["ligand"].x.size(0)
    N_rec = data["receptor"].x.size(0)

    assert out["ligand_pos_pred"].shape == (N_lig, 3)
    assert out["ligand_x"].shape == (N_lig, node_dim)
    assert out["receptor_x"].shape == (N_rec, node_dim)
    assert out["ligand_kp_pos"].shape == (num_keypoints, 3)
    assert out["receptor_kp_pos"].shape == (num_keypoints, 3)
    assert out["R"].shape == (3, 3)
    assert out["t"].shape == (3,)


def test_equibind_rigid_se3_equivariance():
    """
    End-to-end SE(3) behavior:

    If we apply a global rotation+translation to both ligand and receptor
    coordinates in the input, the predicted ligand pose should transform
    accordingly:

        lig_pos_pred_rot ≈ lig_pos_pred @ R^T + t

    Node features remain identical (equivariant coord updates + feature-only
    attention).
    """
    torch.manual_seed(2)

    node_dim = 32
    num_layers = 2
    num_keypoints = 8

    data = build_toy_heterodata(n_lig=6, n_rec=10, node_dim=node_dim)

    model = EquiBindRigid(
        node_dim=node_dim,
        num_layers=num_layers,
        num_keypoints=num_keypoints,
        edge_mlp_dim=32,
        coord_mlp_dim=32,
        cross_attn_dim=32,
    )

    # Forward on original coordinates
    out1 = model(data)
    lig_pos_pred1 = out1["ligand_pos_pred"]
    lig_x1 = out1["ligand_x"]
    rec_x1 = out1["receptor_x"]

    # Apply global SE(3) transform to BOTH ligand and receptor coordinates
    R_global = random_rotation_matrix()     # [3, 3]
    t_global = torch.randn(1, 3)           # [1, 3]

    data_rot = data.clone()
    data_rot["ligand"].pos = data["ligand"].pos @ R_global.T + t_global
    data_rot["receptor"].pos = data["receptor"].pos @ R_global.T + t_global

    out2 = model(data_rot)
    lig_pos_pred2 = out2["ligand_pos_pred"]
    lig_x2 = out2["ligand_x"]
    rec_x2 = out2["receptor_x"]

    # Feature invariance to global SE(3)
    assert torch.allclose(lig_x1, lig_x2, atol=1e-5), (
        "Ligand features should be invariant under global SE(3) transform."
    )
    assert torch.allclose(rec_x1, rec_x2, atol=1e-5), (
        "Receptor features should be invariant under global SE(3) transform."
    )

    # Coordinate equivariance: predicted ligand coordinates should rotate+translate
    lig_pos_pred1_rot = lig_pos_pred1 @ R_global.T + t_global
    assert torch.allclose(lig_pos_pred1_rot, lig_pos_pred2, atol=5e-3), (
        "Predicted ligand coordinates are not equivariant under global SE(3) transform."
    )
