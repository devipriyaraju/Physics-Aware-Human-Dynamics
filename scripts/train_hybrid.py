from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from physics_human_dynamics.grf import (
    physics_grf,
    contact_from_measured_force,
    contact_from_motion,
)
from physics_human_dynamics.hybrid import SplitNet, random_yaw
from physics_human_dynamics.utils import (
    ensure_dir,
    load_config,
    resolve_device,
    seed_everything,
)

def windows(trial, length, stride, device):
    n = len(trial["rel"])
    starts = list(range(0, n - length + 1, stride))
    if not starts:
        return None

    measured_contact = contact_from_measured_force(trial)
    motion_contact = contact_from_motion(trial)
    measured_physics, zmp, total_force = physics_grf(
        trial,
        measured_contact,
    )
    motion_physics, _, _ = physics_grf(
        trial,
        motion_contact,
    )
    arrays = [
        trial["rel"],
        trial["grf"].reshape(n, 6),
        trial["valid"].astype(np.float32),
        total_force.astype(np.float32),
        (
            zmp[:, None] - trial["feet"][:, :, :2]
        ).reshape(n, 4).astype(np.float32),
        measured_physics.reshape(n, 6).astype(np.float32),
        motion_physics.reshape(n, 6).astype(np.float32),
    ]
    return [
        torch.tensor(
            np.stack([array[s:s + length] for s in starts]),
            device=device,
        )
        for array in arrays
    ]

def fit(data, mode, cfg, device):
    hy = cfg["hybrid"]
    seed_everything(cfg["seed"])
    model = SplitNet(
        mode,
        fps=cfg["target_fps"],
        width=hy["width"],
    ).to(device)
    n = len(data["X"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hy["learning_rate"],
        weight_decay=1e-3,
    )
    steps_per_epoch = (n + hy["batch_size"] - 1) // hy["batch_size"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        hy["learning_rate"],
        total_steps=hy["epochs"] * steps_per_epoch,
    )

    for _ in range(hy["epochs"]):
        model.train()
        permutation = torch.randperm(n, device=device)
        for start in range(0, n, hy["batch_size"]):
            batch = permutation[start:start + hy["batch_size"]]
            rotation = random_yaw(len(batch), device)
            rotate3 = lambda v: torch.einsum(
                "bij,bt...j->bt...i",
                rotation,
                v,
            )
            rotate2 = lambda v: torch.einsum(
                "bij,bt...j->bt...i",
                rotation[:, :2, :2],
                v,
            )

            x = rotate3(data["X"][batch])
            y = rotate3(
                data["Y"][batch].view(
                    len(batch),
                    hy["window"],
                    2,
                    3,
                )
            ).reshape(len(batch), hy["window"], 6)
            total_force = rotate3(data["FT"][batch])
            zmp = rotate2(
                data["Z"][batch].view(
                    len(batch),
                    hy["window"],
                    2,
                    2,
                )
            ).reshape(len(batch), hy["window"], 4)

            prediction = model(x, total_force, zmp)
            loss = (
                F.smooth_l1_loss(
                    prediction,
                    y,
                    reduction="none",
                    beta=0.1,
                ).mean(-1)
                * data["V"][batch]
            ).sum() / data["V"][batch].sum().clamp_min(1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()
            scheduler.step()

    return model.eval()

def score(prediction, data):
    valid = data["V"] > 0.5
    pred = prediction[valid].view(-1, 2, 3)
    target = data["Y"][valid].view(-1, 2, 3)
    return {
        "vertical force per foot, RMSE (BW)":
            (pred[..., 2] - target[..., 2])
            .pow(2).mean().sqrt().item(),
        "3D force per foot, RMSE (BW)":
            (pred - target)
            .pow(2).sum(-1).mean().sqrt().item(),
        "total vertical force, RMSE (BW)":
            (
                pred[..., 2].sum(1)
                - target[..., 2].sum(1)
            ).pow(2).mean().sqrt().item(),
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
    hycfg = cfg["hybrid"]

    subjects = sorted({t["subj"] for t in trials})
    data = {}
    names = ["X", "Y", "V", "FT", "Z", "PA", "PB"]
    for subject in subjects:
        chunks = [
            windows(
                trial,
                hycfg["window"],
                hycfg["stride"],
                device,
            )
            for trial in trials
            if trial["subj"] == subject and trial["motion"] != "dog"
        ]
        chunks = [chunk for chunk in chunks if chunk is not None]
        if chunks:
            data[subject] = {
                name: torch.cat(
                    [chunk[i] for chunk in chunks]
                )
                for i, name in enumerate(names)
            }

    modes = {
        "network, direct (no physics)": "direct",
        "hybrid: physics total + learned split": "hybrid",
        "hybrid + physics quantities as input": "hybrid+physics",
    }
    results = {}
    output = ensure_dir(
        Path(cfg["paths"]["results_dir"]) / "06_hybrid_grf"
    )
    checkpoint_dir = ensure_dir(output / "checkpoints")

    for held_out in sorted(data):
        train = {
            key: torch.cat(
                [
                    data[subject][key]
                    for subject in data
                    if subject != held_out
                ]
            )
            for key in data[held_out]
        }
        test = data[held_out]

        results[(held_out, "physics rule, measured contact (upper bound)")] = score(
            test["PA"],
            test,
        )
        results[(held_out, "physics rule, contact from motion")] = score(
            test["PB"],
            test,
        )

        for label, mode in modes.items():
            model = fit(train, mode, cfg, device)
            predictions = []
            with torch.no_grad():
                for start in range(0, len(test["X"]), 1024):
                    predictions.append(
                        model(
                            test["X"][start:start + 1024],
                            test["FT"][start:start + 1024],
                            test["Z"][start:start + 1024],
                        )
                    )
            prediction = torch.cat(predictions)
            results[(held_out, label)] = score(
                prediction,
                test,
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mode": mode,
                    "held_out_subject": held_out,
                    "config": cfg,
                },
                checkpoint_dir / f"{held_out}_{mode.replace('+', '_').replace(' ', '_')}.pt",
            )
        print("held out", held_out)

    frame = pd.DataFrame(results).T.astype(float)
    frame.index.names = ["subject", "method"]
    frame.to_csv(output / "hybrid_all.csv")
    summary = frame.groupby(level="method").agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    print(summary)

if __name__ == "__main__":
    main()
