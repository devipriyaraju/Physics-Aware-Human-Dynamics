from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import torch

from .smplh import axis_angle_to_matrix
from .kinematics import lowpass, derivative
from .inverse_dynamics import body_state, newton_euler
from .constants import G_ACC

def parse_gender(metadata) -> str:
    value = metadata["gender"]
    value = value.item() if value.shape == () else value
    return value.decode() if isinstance(value, bytes) else str(value)

def sole_patches(model, points_per_patch: int = 20) -> torch.Tensor:
    joints = model.joint_regressor @ model.v_template
    dominant = model.weights.argmax(1)
    y = model.v_template[:, 1]
    z = model.v_template[:, 2]
    output = []
    for ankle, toe in [(7, 10), (8, 11)]:
        heel = torch.where(
            (dominant == ankle) & (z < joints[ankle, 2])
        )[0]
        forefoot = torch.where(dominant == toe)[0]
        for index in (heel, forefoot):
            if len(index) < points_per_patch:
                raise ValueError("Too few candidate vertices for a sole patch")
            output.append(index[y[index].argsort()[:points_per_patch]])
    return torch.cat(output)

@torch.no_grad()
def groundlink_fk(
    metadata,
    models: dict,
    reference_model,
    patch: torch.Tensor,
    device: torch.device,
    points_per_patch: int = 20,
):
    model = models.get(parse_gender(metadata), reference_model)
    n = len(metadata["poses"])
    pose = torch.zeros(n, model.n_joints * 3, device=device)
    pose[:, :66] = torch.tensor(
        metadata["poses"][:, :66],
        dtype=torch.float32,
        device=device,
    )

    rest_joints = model.joint_regressor @ model.v_template
    joint_batches = []
    contact_batches = []
    rotation_batches = []

    for start in range(0, n, 4096):
        batch_pose = pose[start:start + 4096]
        rotation = axis_angle_to_matrix(
            batch_pose.view(-1, model.n_joints, 3)
        )
        global_rotation = [rotation[:, 0]]
        global_position = [
            torch.zeros(len(rotation), 3, device=device)
        ]

        for i in range(1, 22):
            parent = model.parents[i]
            global_rotation.append(
                global_rotation[parent] @ rotation[:, i]
            )
            offset = rest_joints[i] - rest_joints[parent]
            global_position.append(
                global_position[parent]
                + (
                    global_rotation[parent] @ offset[:, None]
                ).squeeze(-1)
            )

        joints, vertices = model(
            batch_pose,
            torch.zeros(16, device=device),
            torch.zeros(len(rotation), 3, device=device),
            patch,
        )
        joint_batches.append(torch.stack(global_position, 1))
        contact_batches.append(
            vertices.view(-1, 4, points_per_patch, 3).mean(2)
            - joints[:, :1]
        )
        rotation_batches.append(torch.stack(global_rotation, 1))

    return [
        torch.cat(batch).double().cpu().numpy()
        for batch in (
            joint_batches,
            contact_batches,
            rotation_batches,
        )
    ]

def load_trial(
    force_file: str | Path,
    motion_file: str | Path,
    models: dict,
    reference_model,
    patch: torch.Tensor,
    device: torch.device,
):
    force_file = Path(force_file)
    name = force_file.stem
    force_data = np.load(force_file, allow_pickle=True).item()
    motion_data = np.load(motion_file, allow_pickle=True)

    grf = np.nan_to_num(np.asarray(force_data["GRF"], dtype=np.float64))
    cop = np.nan_to_num(np.asarray(force_data["CoP"], dtype=np.float64))
    if grf.shape[1:] != (2, 3) or cop.shape[1:] != (2, 3):
        raise ValueError(
            f"Unexpected force layout {grf.shape} and {cop.shape}"
        )

    joints, contacts, rotations = groundlink_fk(
        motion_data,
        models,
        reference_model,
        patch,
        device,
    )
    subject = re.search(r"s\d{3}", name)
    if subject is None:
        raise ValueError(f"Could not parse subject from {name}")

    return {
        "name": name,
        "subj": subject.group(),
        "motion": name[14:-2],
        "fps": float(motion_data["mocap_framerate"]),
        "gender": parse_gender(motion_data),
        "trans": motion_data["trans"].astype(np.float64),
        "J": joints,
        "C": contacts,
        "G": rotations,
        "grf_full": grf,
        "cop_full": cop,
    }

def cut_trial(trial: dict, offset: int = 0) -> dict:
    n = min(
        len(trial["trans"]),
        len(trial["grf_full"]) - offset,
    )
    return {
        **trial,
        "trans": trial["trans"][:n],
        "J": trial["J"][:n],
        "C": trial["C"][:n],
        "G": trial["G"][:n],
        "grf": trial["grf_full"][offset:offset + n],
        "cop": trial["cop_full"][offset:offset + n],
    }

def calibrate(trials: list[dict]) -> dict:
    total_vertical = np.concatenate(
        [t["grf"][:, :, 2].sum(1) for t in trials]
    )
    threshold = 0.2 * np.percentile(total_vertical, 75)
    body_weight = np.median(total_vertical[total_vertical > threshold])
    best = None

    for assign in ([0, 1], [1, 0]):
        sole_term = []
        pelvis_z = []
        ground_z = []
        for trial in trials:
            for foot in range(2):
                loaded = (
                    trial["grf"][:, assign[foot], 2]
                    > 0.3 * body_weight
                )
                sole_term.append(
                    -trial["C"][
                        loaded,
                        2 * foot:2 * foot + 2,
                        2,
                    ].min(1)
                )
                pelvis_z.append(trial["trans"][loaded, 2])
                ground_z.append(trial["cop"][loaded, assign[foot], 2])

        sole_term = np.concatenate(sole_term)
        pelvis_z = np.concatenate(pelvis_z)
        z0 = np.median(np.concatenate(ground_z))

        if sole_term.std() > 0.02:
            scale, _ = np.polyfit(sole_term, pelvis_z, 1)
        else:
            scale = 1.0
        if not 0.85 < scale < 1.2:
            scale = 1.0

        offset_z = z0 - np.median(pelvis_z - scale * sole_term)
        xy_residual = []
        for trial in trials:
            for foot in range(2):
                loaded = (
                    trial["grf"][:, assign[foot], 2]
                    > 0.3 * body_weight
                )
                foot_center = trial["C"][
                    loaded,
                    2 * foot:2 * foot + 2,
                    :2,
                ].mean(1)
                xy_residual.append(
                    trial["cop"][loaded, assign[foot], :2]
                    - (
                        trial["trans"][loaded, :2]
                        + scale * foot_center
                    )
                )

        xy_residual = np.concatenate(xy_residual)
        offset_xy = np.median(xy_residual, 0)
        median_error = np.median(
            np.linalg.norm(xy_residual - offset_xy, axis=1)
        )

        candidate = {
            "bw": float(body_weight),
            "assign": list(assign),
            "scale": float(scale),
            "off": np.array(
                [offset_xy[0], offset_xy[1], offset_z]
            ),
            "cop_err": float(median_error),
            "z0": float(z0),
        }
        if best is None or candidate["cop_err"] < best["cop_err"]:
            best = candidate

    return best

def process_trial(
    trial: dict,
    calibration: dict,
    body_parameters: dict,
    table_parameters: dict,
    children: list[list[int]],
    target_fps: int = 30,
    cutoff_hz: float = 6.0,
):
    fps = trial["fps"]
    scale = calibration["scale"]
    n = len(trial["trans"])
    gender = (
        trial["gender"]
        if trial["gender"] in body_parameters
        else next(iter(body_parameters))
    )
    pelvis = trial["trans"] + calibration["off"]

    state = body_state(
        trial["J"],
        trial["G"],
        pelvis,
        scale,
        body_parameters[gender],
        fps,
        cutoff_hz,
    )
    table_state = body_state(
        trial["J"],
        trial["G"],
        pelvis,
        scale,
        table_parameters[gender],
        fps,
        cutoff_hz,
    )

    force_bw = lowpass(
        trial["grf"][:, calibration["assign"]]
        / calibration["bw"],
        fps,
        cutoff_hz,
    )
    external_force = force_bw * G_ACC
    cop = trial["cop"][:, calibration["assign"]]
    external_point = np.where(
        force_bw[:, :, 2:3] > 0.05,
        cop,
        state["Jw"][:, [7, 8]],
    )

    joint_force, joint_moment = newton_euler(
        state,
        external_force,
        external_point,
        children,
    )
    _, table_moment = newton_euler(
        table_state,
        external_force,
        external_point,
        children,
    )

    contact_world = (
        lowpass(pelvis, fps, cutoff_hz)[:, None]
        + scale * lowpass(trial["C"], fps, cutoff_hz)
    )
    feet = contact_world.reshape(n, 2, 2, 3).mean(2)
    sole_velocity = np.linalg.norm(
        derivative(contact_world, fps),
        axis=-1,
    ).reshape(n, 2, 2).min(2)
    sole_height = (
        contact_world[..., 2].reshape(n, 2, 2).min(2)
        - calibration["z0"]
    )

    indices = np.round(
        np.arange(0, n - 1, fps / target_fps)
    ).astype(int)
    as_float32 = lambda x: x[indices].astype(np.float32)

    return {
        "name": trial["name"],
        "subj": trial["subj"],
        "motion": trial["motion"],
        "gender": gender,
        "scale": scale,
        "z0": calibration["z0"],
        "valid": (
            np.linalg.norm(joint_force[:, 0], axis=1)
            < 0.25 * G_ACC
        )[indices],
        "resF": as_float32(
            np.linalg.norm(joint_force[:, 0], axis=1)
        ),
        "resM": as_float32(
            np.linalg.norm(joint_moment[:, 0, :2], axis=1)
        ),
        "resMz": as_float32(
            np.abs(joint_moment[:, 0, 2])
        ),
        "tau": as_float32(joint_moment),
        "tau_table": as_float32(table_moment),
        "grf": as_float32(force_bw),
        "cop": as_float32(cop),
        "com": as_float32(state["com"]),
        "acom": as_float32(state["acom"]),
        "Hdot": as_float32(state["Hdot"]),
        "pel": as_float32(state["Jw"][:, 0]),
        "feet": as_float32(feet),
        "sole_h": as_float32(sole_height),
        "sole_v": as_float32(sole_velocity),
        "rel": as_float32(
            (state["Jw"] - state["Jw"][:, :1]) / scale
        ),
        "G": as_float32(trial["G"]),
    }


def cop_residual(trial: dict, calibration: dict, offset: int):
    cut = cut_trial(trial, offset)
    best = (np.inf, calibration["assign"])
    if len(cut["trans"]) < trial["fps"]:
        return best
    for assign in ([0, 1], [1, 0]):
        distances = []
        for foot in range(2):
            loaded = cut["grf"][:, assign[foot], 2] > 0.3 * calibration["bw"]
            if loaded.sum() == 0:
                continue
            foot_center = cut["C"][
                loaded,
                2 * foot:2 * foot + 2,
                :2,
            ].mean(1)
            predicted = (
                cut["trans"][loaded, :2]
                + calibration["off"][:2]
                + calibration["scale"] * foot_center
            )
            distances.append(
                np.linalg.norm(
                    cut["cop"][loaded, assign[foot], :2] - predicted,
                    axis=1,
                )
            )
        if not distances:
            continue
        distance = np.concatenate(distances)
        if len(distance) > 10 and np.median(distance) < best[0]:
            best = (np.median(distance), assign)
    return best

def align_trial(trial: dict, calibration: dict, step: int = 2):
    error0, assign0 = cop_residual(trial, calibration, 0)
    if error0 <= 0.10:
        return 0, error0, assign0

    max_offset = max(
        1,
        len(trial["grf_full"]) - len(trial["trans"]) + 1,
    )
    candidates = []
    for offset in range(0, max_offset, step):
        error, assign = cop_residual(trial, calibration, offset)
        candidates.append((error, assign, offset))
    error, assign, offset = min(candidates, key=lambda x: x[0])
    return offset, error, assign
