from pathlib import Path
import argparse
import re
import torch

from physics_human_dynamics.utils import load_config, resolve_device, seed_everything
from physics_human_dynamics.smplh import SMPLH
from physics_human_dynamics.anthropometry import (
    segment_parameters,
    winter_table_parameters,
    children_from_parents,
)
from physics_human_dynamics.groundlink import (
    sole_patches,
    load_trial,
    cut_trial,
    calibrate,
    align_trial,
    process_trial,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])

    smpl_dir = Path(cfg["paths"]["smplh_dir"])
    model_paths = {
        p.parent.name: p
        for p in smpl_dir.rglob("model.npz")
    }
    if not model_paths:
        raise FileNotFoundError(
            f"No SMPL-H model.npz files found under {smpl_dir}"
        )
    models = {
        gender: SMPLH(path, device)
        for gender, path in model_paths.items()
    }
    reference = models.get("neutral", next(iter(models.values())))
    patch = sole_patches(reference)
    body_parameters = {
        gender: segment_parameters(model, seed=cfg["seed"])
        for gender, model in models.items()
    }
    table_parameters = {
        gender: winter_table_parameters(model)
        for gender, model in models.items()
    }
    children = children_from_parents(reference.parents[:22])

    force_dir = Path(cfg["paths"]["groundlink_force_dir"])
    motion_dir = Path(cfg["paths"]["groundlink_moshpp_dir"])
    force_files = sorted(force_dir.rglob("*.npy"))
    motion_files = {
        p.name.replace("_stageii.npz", ""): p
        for p in motion_dir.rglob("*_stageii.npz")
    }
    pairs = [
        (f, motion_files[f.stem])
        for f in force_files
        if f.stem in motion_files
    ]
    if not pairs:
        raise FileNotFoundError(
            "No paired GroundLink force and MoSh++ trials were found"
        )

    by_subject = {}
    for force_file, motion_file in pairs:
        name = force_file.stem
        motion = name[14:-2]
        if "ballethigh" in motion or "ballet_high" in motion:
            continue
        if name.startswith("s001") and motion.startswith("idling"):
            continue
        subject = re.search(r"s\d{3}", name).group()
        by_subject.setdefault(subject, []).append(
            (force_file, motion_file)
        )

    processed = []
    calibrations = {}
    for subject, subject_pairs in sorted(by_subject.items()):
        raw = []
        for force_file, motion_file in subject_pairs:
            try:
                raw.append(
                    load_trial(
                        force_file,
                        motion_file,
                        models,
                        reference,
                        patch,
                        device,
                    )
                )
            except Exception as exc:
                print("skip", force_file.name, exc)

        cop_loaded = [
            t["cop_full"][t["grf_full"][:, :, 2] > 0][:, :2]
            for t in raw
        ]
        if cop_loaded:
            import numpy as np
            if np.median(np.abs(np.concatenate(cop_loaded))) > 20:
                for trial in raw:
                    trial["cop_full"] = trial["cop_full"] / 1000.0

        stationary = [
            cut_trial(t)
            for t in raw
            if "walk" not in t["motion"] and "soccer" not in t["motion"]
        ]
        calibrations[subject] = calibrate(
            stationary or [cut_trial(t) for t in raw]
        )

        kept = 0
        for trial in raw:
            offset, error, assign = align_trial(
                trial,
                calibrations[subject],
            )
            if error <= 0.10:
                calibration = {
                    **calibrations[subject],
                    "assign": assign,
                }
                processed.append(
                    process_trial(
                        cut_trial(trial, offset),
                        calibration,
                        body_parameters,
                        table_parameters,
                        children,
                        target_fps=cfg["target_fps"],
                        cutoff_hz=cfg["lowpass_hz"],
                    )
                )
                kept += 1
        print(subject, "kept", kept, "of", len(raw))

    cache = Path(cfg["paths"]["processed_cache"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save((processed, calibrations), cache)
    valid = sum(t["valid"].sum() for t in processed)
    frames = sum(len(t["valid"]) for t in processed)
    print("trials", len(processed))
    print("valid frames percent", 100.0 * valid / frames)

if __name__ == "__main__":
    main()
