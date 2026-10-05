# Checkpoints

`t3d_offline.pt` is the temporal articulated tracker checkpoint used by the tracker-to-physics experiment in the original notebook.

The checkpoint uses the original notebook parameter names. `physics_human_dynamics.tracker.load_tracker_checkpoint` remaps those names to the cleaned module names before strict loading.
