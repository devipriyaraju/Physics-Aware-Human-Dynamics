import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, width: int, kernel: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            width,
            width,
            kernel,
            padding=dilation * (kernel // 2),
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):
        return x + F.gelu(self.norm(self.conv(x)))

class SplitNet(nn.Module):
    def __init__(self, mode: str, fps: int = 30, width: int = 256):
        super().__init__()
        if mode not in {"direct", "hybrid", "hybrid+physics"}:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.fps = fps
        extra = 7 if mode == "hybrid+physics" else 0
        self.input = nn.Conv1d(22 * 6 + extra, width, 1)
        self.body = nn.Sequential(
            *[ResidualBlock(width, 3, d) for d in (1, 2, 4, 8)]
        )
        self.head = nn.Conv1d(width, 6 if mode == "direct" else 3, 1)

    def forward(self, joints, total_force, zmp_relative):
        velocity = torch.zeros_like(joints)
        velocity[:, 1:] = (
            joints[:, 1:] - joints[:, :-1]
        ) * self.fps
        velocity[:, 0] = velocity[:, 1]

        features = torch.cat([joints, velocity], dim=-1).flatten(2)
        if self.mode == "hybrid+physics":
            features = torch.cat(
                [features, total_force, zmp_relative.clamp(-2, 2)],
                dim=-1,
            )

        output = self.head(
            self.body(self.input(features.transpose(1, 2)))
        ).transpose(1, 2)

        if self.mode == "direct":
            return output

        left_share = torch.sigmoid(output[..., :1])
        horizontal_pair = 0.1 * output[..., 1:3]
        vertical = total_force[..., 2:].clamp_min(0)

        left = torch.cat(
            [
                left_share * total_force[..., :2] + horizontal_pair,
                left_share * vertical,
            ],
            dim=-1,
        )
        right = torch.cat(
            [
                (1.0 - left_share) * total_force[..., :2] - horizontal_pair,
                (1.0 - left_share) * vertical,
            ],
            dim=-1,
        )
        return torch.cat([left, right], dim=-1)

def random_yaw(batch: int, device: torch.device) -> torch.Tensor:
    angle = torch.rand(batch, device=device) * 2 * np.pi
    c = angle.cos()
    s = angle.sin()
    z = torch.zeros(batch, device=device)
    o = torch.ones(batch, device=device)
    return torch.stack(
        [c, -s, z, s, c, z, z, z, o],
        dim=-1,
    ).view(batch, 3, 3)
