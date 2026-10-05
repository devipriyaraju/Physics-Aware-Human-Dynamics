from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

from physics_human_dynamics.constants import JOINT_GROUPS
from physics_human_dynamics.utils import load_config, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    trials, _ = torch.load(
        cfg["paths"]["processed_cache"],
        weights_only=False,
    )

    mesh = np.concatenate(
        [t["tau"][t["valid"]] for t in trials]
    )
    table = np.concatenate(
        [t["tau_table"][t["valid"]] for t in trials]
    )

    rows = {}
    for name, joints in JOINT_GROUPS.items():
        typical = np.linalg.norm(mesh[:, joints], axis=-1).mean()
        difference = np.linalg.norm(
            mesh[:, joints] - table[:, joints],
            axis=-1,
        ).mean()
        rows[name] = {
            "typical size, mesh parameters (N m/kg)": typical,
            "difference mesh vs table (N m/kg)": difference,
            "difference as percent of size": 100.0 * difference / typical,
        }

    output = ensure_dir(
        Path(cfg["paths"]["results_dir"]) / "02_inverse_dynamics"
    )
    frame = pd.DataFrame(rows).T.round(3)
    frame.to_csv(output / "body_parameter_sensitivity.csv")
    print(frame)

if __name__ == "__main__":
    main()
