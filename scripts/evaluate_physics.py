from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

from physics_human_dynamics.utils import load_config, ensure_dir
from physics_human_dynamics.grf import (
    physics_grf,
    contact_from_measured_force,
    contact_from_motion,
)

def evaluate(trials, contact_fn, friction_coefficient):
    foot_errors = []
    total_errors = []
    zmp_errors = []
    violations = np.zeros(5)

    for trial in trials:
        valid = trial["valid"]
        contact = contact_fn(trial)
        per_foot, zmp, total_force = physics_grf(trial, contact)
        measured = trial["grf"]

        foot_errors.append(
            (per_foot[valid, :, 2] - measured[valid, :, 2]).ravel()
        )
        total_errors.append(
            total_force[valid, 2] - measured[valid, :, 2].sum(1)
        )

        loaded = valid & (measured[:, :, 2].sum(1) > 0.3)
        measured_cop = (
            trial["cop"][:, :, :2] * measured[:, :, 2:3]
        ).sum(1) / np.maximum(
            measured[:, :, 2:3].sum(1),
            1e-6,
        )
        zmp_errors.append(
            np.linalg.norm(zmp[loaded] - measured_cop[loaded], axis=1)
        )

        any_contact = contact.any(1)
        foot_distance = np.linalg.norm(
            zmp[:, None] - trial["feet"][:, :, :2],
            axis=-1,
        ).min(1)
        violations += [
            valid.sum(),
            (valid & (total_force[:, 2] < 0)).sum(),
            (
                valid
                & any_contact
                & (
                    np.linalg.norm(total_force[:, :2], axis=1)
                    > friction_coefficient
                    * np.maximum(total_force[:, 2], 0)
                )
            ).sum(),
            (
                valid
                & ~any_contact
                & (np.linalg.norm(total_force, axis=1) > 0.2)
            ).sum(),
            (
                valid
                & any_contact
                & ~(contact[:, 0] & contact[:, 1])
                & (foot_distance > 0.20)
            ).sum(),
        ]

    rms = lambda arrays: float(
        np.sqrt(np.mean(np.concatenate(arrays) ** 2))
    )
    return {
        "vertical force per foot, RMSE (BW)": rms(foot_errors),
        "total vertical force, RMSE (BW)": rms(total_errors),
        "zero-moment point vs measured CoP, median (cm)":
            100.0 * float(np.median(np.concatenate(zmp_errors))),
        "frames needing a pulling force (%)":
            100.0 * violations[1] / violations[0],
        "frames outside the friction cone (%)":
            100.0 * violations[2] / violations[0],
        "no contact but force > 0.2 BW (%)":
            100.0 * violations[3] / violations[0],
        "single support, ZMP > 20 cm from the foot (%)":
            100.0 * violations[4] / violations[0],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    trials, _ = torch.load(
        cfg["paths"]["processed_cache"],
        weights_only=False,
    )

    both = lambda d: np.ones_like(
        d["grf"][:, :, 2],
        dtype=bool,
    )
    table = pd.DataFrame({
        "physics, measured contact": evaluate(
            trials,
            contact_from_measured_force,
            cfg["friction_coefficient"],
        ),
        "physics, contact from motion": evaluate(
            trials,
            contact_from_motion,
            cfg["friction_coefficient"],
        ),
        "physics, both feet always down": evaluate(
            trials,
            both,
            cfg["friction_coefficient"],
        ),
    }).round(3)

    output = ensure_dir(
        Path(cfg["paths"]["results_dir"]) / "03_physics_grf"
    )
    table.to_csv(output / "physics_grf.csv")
    print(table)

if __name__ == "__main__":
    main()
