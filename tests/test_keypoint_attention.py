# tests/test_keypoint_attention.py

import torch

from equibind_pyg.layers.keypoint_attention import KeypointAttention


def random_rotation_matrix() -> torch.Tensor:
    """
    Generate a random proper rotation matrix R ∈ SO(3) using QR decomposition.
    """
    Q = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(Q)
    if torch.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


def test_keypoint_attention_shapes():
    """
    Basic sanity: KeypointAttention runs and returns expected shapes.
    """
    torch.manual_seed(0)

    N = 10            # number of nodes
    d = 32            # node feature dim
    K = 4             # number of keypoints
    out_dim = 64      # keypoint feature dim

    x = torch.randn(N, d)
    pos = torch.randn(N, 3)

    kp_attn = KeypointAttention(
        num_keypoints=K,
        node_dim=d,
        attn_dim=d,
        out_dim=out_dim,
    )

    kp_pos, kp_feat, attn = kp_attn(x, pos)

    assert kp_pos.shape == (K, 3)
    assert kp_feat.shape == (K, out_dim)
    assert attn.shape == (K, N)


def test_keypoint_attention_se3_behavior():
    """
    Test SE(3) behavior:

    - Features (kp_feat) depend only on x, so they should be invariant to
      global rotation/translation of coordinates.
    - Positions (kp_pos) are convex combinations of input pos, so they
      should transform equivariantly:
          kp_pos_rotated ≈ kp_pos @ R^T + t
    """
    torch.manual_seed(1)

    N = 12
    d = 16
    K = 3

    x = torch.randn(N, d)
    pos = torch.randn(N, 3)

    kp_attn = KeypointAttention(
        num_keypoints=K,
        node_dim=d,
        attn_dim=d,
        out_dim=d,
    )

    # Forward on original
    kp_pos1, kp_feat1, attn1 = kp_attn(x, pos)

    # Apply a random global SE(3) transform to coordinates
    R = random_rotation_matrix()        # [3, 3]
    t = torch.randn(1, 3)               # [1, 3]

    pos_rot = pos @ R.T + t             # [N, 3]

    kp_pos2, kp_feat2, attn2 = kp_attn(x, pos_rot)

    # Attention weights depend only on x, so they should be identical
    assert torch.allclose(attn1, attn2, atol=1e-6), (
        "Attention weights should be invariant to global SE(3) transform."
    )

    # Keypoint features depend only on x, so they should be identical
    assert torch.allclose(kp_feat1, kp_feat2, atol=1e-6), (
        "Keypoint features should be invariant to global SE(3) transform."
    )

    # Keypoint positions should transform equivariantly
    kp_pos1_rot = kp_pos1 @ R.T + t     # [K, 3]
    assert torch.allclose(kp_pos1_rot, kp_pos2, atol=1e-5), (
        "Keypoint positions are not equivariant under global SE(3) transform."
    )
