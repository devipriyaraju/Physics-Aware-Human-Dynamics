from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, width: int, kernel: int, dilation: int, causal: bool):
        super().__init__()
        if causal:
            self.pad = (dilation * (kernel - 1), 0)
        else:
            self.pad = (dilation * (kernel // 2), dilation * (kernel // 2))
        self.conv = nn.Conv1d(width, width, kernel, dilation=dilation)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x, self.pad)).transpose(1, 2)
        y = F.gelu(self.norm(y)).transpose(1, 2)
        return x + y


class Tracker(nn.Module):
    def __init__(self, dim: int = 3, mode: str = "offline", width: int = 256):
        super().__init__()
        kernel = 1 if mode == "frame" else 3
        self.input = nn.Conv1d(22 * (dim + 1), width, 1)
        self.body = nn.Sequential(
            *[
                TemporalBlock(
                    width,
                    kernel,
                    dilation if kernel > 1 else 1,
                    mode == "causal",
                )
                for dilation in (1, 2, 4, 8, 16)
            ]
        )
        self.rotation = nn.Conv1d(width, 22 * 6, 1)
        self.contact = nn.Conv1d(width, 4, 1)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor):
        features = torch.cat(
            [observed * mask[..., None], mask[..., None]],
            dim=-1,
        )
        h = self.body(self.input(features.flatten(2).transpose(1, 2)))
        x = self.rotation(h).transpose(1, 2)
        x = x.reshape(*observed.shape[:2], 22, 6)
        b1 = F.normalize(x[..., :3], dim=-1)
        raw_b2 = x[..., 3:]
        b2 = F.normalize(
            raw_b2 - (b1 * raw_b2).sum(-1, keepdim=True) * b1,
            dim=-1,
        )
        rotation = torch.stack(
            [b1, b2, torch.cross(b1, b2, dim=-1)],
            dim=-1,
        )
        contact = self.contact(h).transpose(1, 2)
        return rotation, contact


def _extract_state_dict(payload) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("Tracker checkpoint must contain a state dictionary")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise TypeError("Tracker state_dict entry is not a mapping")
    return state


def _remap_notebook_keys(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the original notebook checkpoint names to the modular class names."""
    remapped = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("inp."):
            new_key = "input." + new_key[len("inp."):]
        elif new_key.startswith("rot."):
            new_key = "rotation." + new_key[len("rot."):]
        elif new_key.startswith("con."):
            new_key = "contact." + new_key[len("con."):]
        new_key = new_key.replace(".c.", ".conv.")
        new_key = new_key.replace(".n.", ".norm.")
        remapped[new_key] = value
    return remapped


def load_tracker_checkpoint(
    model: Tracker,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
    map_location: str | torch.device | None = None,
) -> Tracker:
    """Load either the original notebook checkpoint or the modular format."""
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(
            checkpoint,
            map_location=map_location,
            weights_only=False,
        )
    else:
        payload = checkpoint
    state = _remap_notebook_keys(_extract_state_dict(payload))
    model.load_state_dict(state, strict=True)
    return model
