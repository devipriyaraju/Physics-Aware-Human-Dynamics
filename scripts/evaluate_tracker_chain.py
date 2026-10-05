from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

from physics_human_dynamics.grf import physics_grf, contact_from_measured_force
from physics_human_dynamics.inverse_dynamics import body_state
from physics_human_dynamics.tracker import Tracker, load_tracker_checkpoint
from physics_human_dynamics.utils import load_config, resolve_device, ensure_dir
from physics_human_dynamics.anthropometry import segment_parameters
from physics_human_dynamics.smplh import SMPLH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])

    checkpoint_path = Path(cfg["paths"]["tracker_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Tracker checkpoint not found: {checkpoint_path}"
        )

    trials, _ = torch.load(
        cfg["paths"]["processed_cache"],
        weights_only=False,
    )

    model_paths = {
        p.parent.name: p
        for p in Path(cfg["paths"]["smplh_dir"]).rglob("model.npz")
    }
    models = {
        gender: SMPLH(path, device)
        for gender, path in model_paths.items()
    }
    reference = models.get("neutral", next(iter(models.values())))
    body_parameters = {
        gender: segment_parameters(model, seed=cfg["seed"])
        for gender, model in models.items()
    }

    tracker = Tracker().to(device)
    load_tracker_checkpoint(tracker, checkpoint_path, map_location=device)
    tracker.eval()

    parent = reference.parents[:22]
    rest = reference.joint_regressor @ reference.v_template
    offsets = torch.stack([
        rest[k] - rest[parent[k]] if k else torch.zeros(3, device=device)
        for k in range(22)
    ])[None]

    force_errors = {"true pose, 30 Hz": [], "tracked pose": []}
    joint_errors = []

    for index, trial in enumerate(trials):
        n = len(trial["rel"])
        if n < 96:
            continue

        torch.manual_seed(index)
        observed = torch.tensor(trial["rel"], device=device)[None]
        mask = torch.ones(1, n, 22, device=device)
        for start in range(30, n - 30, 90):
            mask[:, start:start + 30, [1, 4, 7, 10]] = 0

        with torch.no_grad():
            rotation, _ = tracker(
                observed + torch.randn_like(observed) * 0.015,
                mask,
            )

            global_rotation = [rotation[:, :, 0]]
            global_position = [
                torch.zeros(1, n, 3, device=device)
            ]
            for joint in range(1, 22):
                p = parent[joint]
                global_rotation.append(
                    global_rotation[p] @ rotation[:, :, joint]
                )
                global_position.append(
                    global_position[p]
                    + (
                        global_rotation[p]
                        @ offsets[:, None, joint, :, None]
                    ).squeeze(-1)
                )

        tracked_rel = torch.stack(global_position, 2)[0].cpu().numpy()
        tracked_rot = torch.stack(global_rotation, 2)[0].cpu().numpy()
        joint_errors.append(
            1000.0 * np.linalg.norm(
                tracked_rel - trial["rel"],
                axis=-1,
            ).mean()
        )

        valid = trial["valid"].copy()
        valid[:5] = False
        valid[-5:] = False
        contact = contact_from_measured_force(trial)

        for label, rel, rot in [
            ("true pose, 30 Hz", trial["rel"], trial["G"]),
            ("tracked pose", tracked_rel, tracked_rot),
        ]:
            gender = trial["gender"] if trial["gender"] in body_parameters else next(iter(body_parameters))
            state = body_state(
                rel.astype(np.float64),
                rot.astype(np.float64),
                trial["pel"].astype(np.float64),
                trial["scale"],
                body_parameters[gender],
                cfg["target_fps"],
            )
            state_trial = {**trial, "com": state["com"], "acom": state["acom"], "Hdot": state["Hdot"]}
            per_foot, _, total = physics_grf(state_trial, contact)
            force_errors[label].append([
                ((per_foot[valid, :, 2] - trial["grf"][valid, :, 2]) ** 2).sum(),
                2 * valid.sum(),
                ((total[valid, 2] - trial["grf"][valid, :, 2].sum(1)) ** 2).sum(),
                valid.sum(),
            ])

    rows = {}
    for label, values in force_errors.items():
        array = np.asarray(values)
        rows[label] = {
            "vertical force per foot, RMSE (BW)": np.sqrt(array[:, 0].sum() / array[:, 1].sum()),
            "total vertical force, RMSE (BW)": np.sqrt(array[:, 2].sum() / array[:, 3].sum()),
        }

    output = ensure_dir(Path(cfg["paths"]["results_dir"]) / "05_tracker_chain")
    pd.DataFrame(rows).T.round(3).to_csv(output / "tracker_chain.csv")
    pd.DataFrame({"tracker joint error on GroundLink motion, mm": [np.mean(joint_errors)]}).to_csv(
        output / "tracker_error.csv",
        index=False,
    )
    print(pd.DataFrame(rows).T.round(3))
    print("tracker joint error, mm", np.mean(joint_errors))

if __name__ == "__main__":
    main()
