import numpy as np
import torch
import torch.nn.functional as F
from .constants import G_ACC, GRAVITY_VECTOR

DEFAULT_WEIGHTS = {
    "position": 0.01,
    "velocity": 0.05,
    "jerk": 2e-6,
    "foot_slide": 1.0,
    "foot_height": 20.0,
    "penetration": 100.0,
    "flight": 2e-3,
    "pulling_force": 2e-3,
    "friction": 2e-3,
}

def corrupt_root(pelvis: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    n = len(pelvis)
    drift = torch.cumsum(torch.randn(3, n, generator=generator), -1)
    drift = F.avg_pool1d(
        F.pad(drift[None], (7, 7), mode="replicate"),
        15,
        1,
    )[0]
    drift = drift - drift[:, :1]
    drift = drift / drift.norm(dim=0).max() * 0.10
    noise = torch.randn(n, 3, generator=generator) * 0.003
    return pelvis + drift.T + noise

def refine_root(
    trial: dict,
    observed_root: torch.Tensor,
    device: torch.device,
    use_contact: bool = True,
    use_dynamics: bool = True,
    fps: int = 30,
    steps: int = 300,
    learning_rate: float = 0.02,
    friction_coefficient: float = 0.8,
    weights: dict | None = None,
):
    w = DEFAULT_WEIGHTS if weights is None else weights
    as_tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    dt = 1.0 / fps
    gravity = as_tensor(GRAVITY_VECTOR)

    qcom = as_tensor(trial["com"] - trial["pel"])
    qfeet = as_tensor(trial["feet"] - trial["pel"][:, None])
    contact = as_tensor(trial["grf"][:, :, 2] > 0.05)
    ground_z = trial["z0"]
    qlow = as_tensor(
        trial["sole_h"] + trial["z0"] - trial["pel"][:, 2:3]
    )

    observed_root = observed_root.to(device)
    observed_velocity = torch.cat(
        [
            torch.zeros(1, 3, device=device),
            (observed_root[1:] - observed_root[:-1]) / dt,
        ]
    )
    velocity = observed_velocity.clone().requires_grad_(True)
    offset = torch.zeros(3, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([velocity, offset], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda k: 1.0 - k / steps,
    )

    def path():
        return (
            observed_root[0]
            + offset
            + torch.cumsum(velocity, 0) * dt
        )

    for _ in range(steps):
        root = path()
        com = root + qcom
        acc = (com[2:] - 2 * com[1:-1] + com[:-2]) / dt ** 2
        feet = root[:, None] + qfeet
        foot_velocity = (feet[1:] - feet[:-1]) / dt
        any_contact = contact.max(1).values[1:-1]
        lowest = root[:, 2:3] + qlow

        loss = (
            w["position"] * ((root - observed_root) ** 2).sum(-1).mean()
            + w["velocity"] * ((velocity - observed_velocity) ** 2).sum(-1).mean()
            + w["jerk"] * ((acc[1:] - acc[:-1]) ** 2).sum(-1).mean()
        )

        if use_contact:
            loss = loss + (
                w["foot_slide"]
                * (contact[1:, :, None] * foot_velocity ** 2)
                .sum(-1)
                .mean()
                + w["foot_height"]
                * (contact * (lowest - ground_z) ** 2).mean()
                + w["penetration"]
                * torch.relu(ground_z - 0.02 - lowest).pow(2).mean()
            )

        if use_dynamics:
            upward = acc[:, 2] + G_ACC
            loss = loss + (
                w["flight"]
                * (
                    (1.0 - any_contact)[:, None]
                    * (acc - gravity) ** 2
                ).sum(-1).mean()
                + w["pulling_force"]
                * (
                    any_contact * torch.relu(-upward) ** 2
                ).mean()
                + w["friction"]
                * (
                    any_contact
                    * torch.relu(
                        acc[:, :2].norm(dim=-1)
                        - friction_coefficient * upward.clamp_min(0)
                    ) ** 2
                ).mean()
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return path().detach().cpu()
