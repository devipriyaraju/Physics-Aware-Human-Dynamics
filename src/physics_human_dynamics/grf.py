import numpy as np
from .constants import G_ACC, GRAVITY_VECTOR

def physics_grf(trial: dict, contact: np.ndarray):
    total_force = (trial["acom"] - GRAVITY_VECTOR) / G_ACC
    hdot = trial["Hdot"] / G_ACC
    rz = trial["z0"] - trial["com"][:, 2]
    vertical = np.maximum(total_force[:, 2], 1e-3)

    zmp = trial["com"][:, :2] + np.stack(
        [
            (rz * total_force[:, 0] - hdot[:, 1]) / vertical,
            (hdot[:, 0] + rz * total_force[:, 1]) / vertical,
        ],
        axis=1,
    )

    left = trial["feet"][:, 0, :2]
    right = trial["feet"][:, 1, :2]
    delta = left - right
    weight = np.clip(
        ((zmp - right) * delta).sum(1)
        / np.maximum((delta ** 2).sum(1), 1e-6),
        0,
        1,
    )
    left_weight = np.where(
        contact[:, 0] & contact[:, 1],
        weight,
        contact[:, 0].astype(float),
    )

    per_foot = np.stack(
        [
            left_weight[:, None] * total_force,
            (1.0 - left_weight)[:, None] * total_force,
        ],
        axis=1,
    )
    per_foot[~contact.any(1)] = 0
    return per_foot, zmp, total_force

def contact_from_measured_force(trial: dict) -> np.ndarray:
    return trial["grf"][:, :, 2] > 0.05

def contact_from_motion(trial: dict) -> np.ndarray:
    return (trial["sole_h"] < 0.05) & (trial["sole_v"] < 0.5)
