from pathlib import Path
import numpy as np
import torch

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = (axis_angle / angle).unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).view(*axis_angle.shape[:-1], 3, 3)
    s = angle.sin()[..., None]
    c = angle.cos()[..., None]
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return eye + s * k + (1.0 - c) * (k @ k)

class SMPLH:
    def __init__(self, model_path: str | Path, device: torch.device, n_betas: int = 16):
        raw = np.load(model_path, allow_pickle=True)
        def dense(array):
            return np.asarray(array.item().todense()) if array.dtype == object else np.asarray(array)
        def tensor(array):
            return torch.tensor(dense(array), dtype=torch.float32, device=device)

        self.device = device
        self.v_template = tensor(raw["v_template"])
        self.shape_dirs = tensor(raw["shapedirs"])[..., :n_betas]
        self.pose_dirs = tensor(raw["posedirs"])
        self.joint_regressor = tensor(raw["J_regressor"])
        self.weights = tensor(raw["weights"])
        parents = raw["kintree_table"][0].astype(np.int64)
        parents[0] = -1
        self.parents = parents.tolist()
        self.n_joints = len(parents)

    def __call__(self, poses, betas, trans, vertex_indices=None):
        batch = len(poses)
        n_betas = min(len(betas), self.shape_dirs.shape[-1])
        v_shaped = self.v_template + self.shape_dirs[..., :n_betas] @ betas[:n_betas]
        joints = self.joint_regressor @ v_shaped
        rotation = axis_angle_to_matrix(poses.view(batch, self.n_joints, 3))

        global_rot = [rotation[:, 0]]
        global_pos = [joints[0].expand(batch, 3)]
        for i in range(1, self.n_joints):
            parent = self.parents[i]
            global_rot.append(global_rot[parent] @ rotation[:, i])
            offset = joints[i] - joints[parent]
            global_pos.append(
                global_pos[parent]
                + (global_rot[parent] @ offset[:, None]).squeeze(-1)
            )
        global_rot = torch.stack(global_rot, 1)
        global_pos = torch.stack(global_pos, 1)

        if vertex_indices is None:
            vertex_indices = torch.arange(len(self.v_template), device=self.device)

        pose_feature = (
            rotation[:, 1:] - torch.eye(3, device=self.device)
        ).reshape(batch, -1)
        v_posed = (
            v_shaped[vertex_indices]
            + (self.pose_dirs[vertex_indices] @ pose_feature.T).permute(2, 0, 1)
        )
        translation = global_pos - (
            global_rot @ joints[None, :, :, None]
        ).squeeze(-1)
        weights = self.weights[vertex_indices]
        rv = torch.einsum("vj,bjmn->bvmn", weights, global_rot)
        tv = torch.einsum("vj,bjm->bvm", weights, translation)
        vertices = (rv @ v_posed[..., None]).squeeze(-1) + tv
        return global_pos + trans[:, None], vertices + trans[:, None]
