from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

from physics_human_dynamics.constants import G_ACC, GRAVITY_VECTOR
from physics_human_dynamics.refinement import corrupt_root, refine_root
from physics_human_dynamics.utils import (
    ensure_dir,
    load_config,
    resolve_device,
)

MOTIONS = [
    "hopping",
    "jumpingjack",
    "balletsmalljump",
    "squat",
    "lambadadance",
    "soccerkick",
    "tennisserve",
    "tennisgroundstroke",
    "walk",
    "walk_00",
    "taichi",
]

def score(trial, root, fps):
    pelvis = torch.tensor(trial["pel"])
    com = (
        root + torch.tensor(trial["com"] - trial["pel"])
    ).numpy()
    total_force = (
        np.gradient(
            np.gradient(com, 1.0 / fps, axis=0),
            1.0 / fps,
            axis=0,
        )
        - GRAVITY_VECTOR
    ) / G_ACC

    valid = trial["valid"].copy()
    valid[:3] = False
    valid[-3:] = False
    measured_total = trial["grf"].sum(1)
    flight = valid & (measured_total[:, 2] < 0.05)

    return np.array([
        (root - pelvis).norm(dim=-1).mean().item() * 100.0,
        ((total_force[valid] - measured_total[valid]) ** 2)
        .sum(-1)
        .sum(),
        valid.sum(),
        np.linalg.norm(total_force[flight], axis=1).sum(),
        flight.sum(),
    ])

def summarize(values):
    array = np.asarray(values)
    return {
        "root error (cm)": array[:, 0].mean(),
        "total force error, RMSE (BW)":
            np.sqrt(array[:, 1].sum() / array[:, 2].sum()),
        "force during flight, mean (BW)":
            array[:, 3].sum() / max(array[:, 4].sum(), 1),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    trials, _ = torch.load(
        cfg["paths"]["processed_cache"],
        weights_only=False,
    )

    selected = [
        t
        for t in trials
        if t["motion"] in MOTIONS and len(t["pel"]) >= 60
    ]
    methods = {
        "corrupted": None,
        "smoothing only": (False, False),
        "smoothing + contact": (True, False),
        "smoothing + contact + dynamics": (True, True),
    }
    accum = {name: [] for name in methods}
    accum["true root (reference)"] = []

    for i, trial in enumerate(selected):
        observed = corrupt_root(
            torch.tensor(trial["pel"]),
            i,
        )
        outputs = {}
        for name, flags in methods.items():
            if flags is None:
                outputs[name] = observed
            else:
                outputs[name] = refine_root(
                    trial,
                    observed,
                    device,
                    use_contact=flags[0],
                    use_dynamics=flags[1],
                    fps=cfg["target_fps"],
                    steps=cfg["refinement"]["steps"],
                    learning_rate=cfg["refinement"]["learning_rate"],
                    friction_coefficient=cfg["friction_coefficient"],
                )
        outputs["true root (reference)"] = torch.tensor(trial["pel"])
        for name, root in outputs.items():
            accum[name].append(
                score(trial, root, cfg["target_fps"])
            )

    table = pd.DataFrame(
        {name: summarize(values) for name, values in accum.items()}
    ).T.round(3)
    output = ensure_dir(
        Path(cfg["paths"]["results_dir"]) / "04_root_refinement"
    )
    table.to_csv(output / "refinement.csv")
    print(table)

if __name__ == "__main__":
    main()
