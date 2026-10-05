from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "00_notebook_reference"


def close(value: float, target: float, tolerance: float = 0.0015) -> None:
    if abs(value - target) > tolerance:
        raise AssertionError(f"Expected {target}, got {value}")


def main() -> None:
    physics = pd.read_csv(REFERENCE / "physics_grf.csv", index_col=0)
    close(
        physics.loc[
            "total vertical force, RMSE (BW)",
            "physics, measured contact",
        ],
        0.048,
    )
    close(
        physics.loc[
            "zero-moment point vs measured CoP, median (cm)",
            "physics, measured contact",
        ],
        3.026,
    )

    hybrid = pd.read_csv(REFERENCE / "hybrid_all.csv")
    metric = "vertical force per foot, RMSE (BW)"
    means = hybrid.groupby("method")[metric].mean()
    close(means["network, direct (no physics)"], 0.072, 0.002)
    close(means["physics rule, contact from motion"], 0.090, 0.002)
    close(means["hybrid: physics total + learned split"], 0.059, 0.002)
    close(means["hybrid + physics quantities as input"], 0.056, 0.002)

    table = hybrid.pivot(index="subject", columns="method", values=metric)
    hybrid_basic = table["hybrid: physics total + learned split"]
    if not (hybrid_basic < table["network, direct (no physics)"]).all():
        raise AssertionError("Basic hybrid does not beat the direct network on every subject")
    if not (hybrid_basic < table["physics rule, contact from motion"]).all():
        raise AssertionError("Basic hybrid does not beat motion-contact physics on every subject")

    tracker = pd.read_csv(REFERENCE / "tracker_chain.csv", index_col=0)
    close(tracker.loc["true pose, 30 Hz", metric], 0.052)
    close(tracker.loc["tracked pose", metric], 0.060)

    print("Reference result verification passed")
    print("subjects", hybrid["subject"].nunique())
    print("basic hybrid mean", round(means["hybrid: physics total + learned split"], 3))
    print("direct network mean", round(means["network, direct (no physics)"], 3))
    print("motion-contact physics mean", round(means["physics rule, contact from motion"], 3))


if __name__ == "__main__":
    main()
