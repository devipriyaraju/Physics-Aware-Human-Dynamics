import torch
from physics_human_dynamics.hybrid import SplitNet

def test_hybrid_preserves_total_force():
    model = SplitNet("hybrid", width=32)
    joints = torch.randn(2, 16, 22, 3)
    total = torch.randn(2, 16, 3)
    total[..., 2] = total[..., 2].abs()
    zmp = torch.randn(2, 16, 4)
    pred = model(joints, total, zmp).view(2, 16, 2, 3)
    assert torch.allclose(
        pred.sum(2),
        total,
        atol=1e-5,
        rtol=1e-5,
    )
