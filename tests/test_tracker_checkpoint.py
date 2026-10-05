from pathlib import Path

from physics_human_dynamics.tracker import Tracker, load_tracker_checkpoint


def test_original_notebook_tracker_checkpoint_loads_strictly():
    checkpoint = Path("checkpoints/t3d_offline.pt")
    if not checkpoint.exists():
        return
    model = Tracker()
    load_tracker_checkpoint(model, checkpoint, map_location="cpu")
