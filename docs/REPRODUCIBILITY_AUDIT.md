# Reproducibility Audit

## Artifacts validated

The repository was checked against the original experiment notebook and the supplied Compute Canada artifacts.

### Result archives

`results.zip` and `results(1).zip` are byte-for-byte identical.

SHA-256:

```text
3cd568632ac056ca4dda13cf586b25dcf713d0ee6c9b893d6d0554a9c2eac231
```

The archive contains the original notebook outputs:

- `full_body_dynamics.csv`
- `hybrid_all.csv`
- `physics_grf.csv`
- `physics_grf_example.png`
- `refinement.csv`
- `refinement_example.png`
- `tracker_chain.csv`

The CSV values match the notebook outputs used by the README.

## Claims supported by the notebook and result artifacts

### Dataset and preprocessing

- 361 GroundLink force files were present before pairing.
- 396 MoSh++ motion files were present in the original notebook run.
- 357 force and motion trials paired directly.
- 336 trials were retained for the dynamics experiments.
- 83.8 percent of frames passed the dynamics consistency filter.
- The 336 retained trials span 7 subjects.

### Physics-only force estimation

- Total vertical force RMSE is 0.048 BW.
- Median ZMP error relative to measured center of pressure is 3.026 cm.
- Per-foot vertical force RMSE is 0.083 BW with measured contact.
- Per-foot vertical force RMSE is 0.105 BW when contact is inferred from motion.
- Motion-derived contact agrees with measured contact on 96.7 percent of valid foot labels.

### Hybrid force model

Across 7 leave-one-subject-out folds:

- Direct temporal network: 0.072 BW mean per-foot vertical force RMSE.
- Physics rule using motion-derived contact: 0.090 BW.
- Hybrid physics total plus learned split: 0.059 BW.
- Hybrid with physics quantities as network input: 0.056 BW.
- The 0.059 BW hybrid beats the direct network on all 7 held-out subjects.
- The 0.059 BW hybrid beats the motion-contact physics rule on all 7 held-out subjects.

The relative improvements for the 0.059 BW hybrid are about 18 percent against the direct network and 34 percent against the motion-contact physics rule.

### Joint torque sensitivity

Mesh-derived body parameters were compared with anthropometric table parameters. Mean moment differences as a percentage of the mesh-derived moment size are:

- ankle: 1.1 percent
- knee: 2.9 percent
- hip: 4.5 percent
- lower back: 25.2 percent
- shoulder: 36.9 percent
- elbow: 52.6 percent

This supports the resume statement that leg torques change by less than 5 percent while lower-back and arm torques change by roughly 25 to 53 percent.

### Tracker-to-physics chain

The tracker stress test uses 15 mm observation noise and hides the left leg for one second out of every three.

- Tracker joint error: 36.1 mm.
- Reference pose per-foot vertical force RMSE: 0.052 BW.
- Tracked pose per-foot vertical force RMSE: 0.060 BW.
- Reference pose total vertical force RMSE: 0.046 BW.
- Tracked pose total vertical force RMSE: 0.048 BW.

The per-foot force degradation from 0.052 to 0.060 BW is about 15 percent.

### Root refinement

The dynamics-aware root objective reduces implied total-force error from 0.305 BW for the corrupted path to 0.177 BW. Root position error changes from 5.709 cm to 6.022 cm, so this experiment should not be described as improving root-position accuracy.

## Supplied raw assets

### GroundLink force archive

The supplied `force.zip` contains seven subject ZIP files. Recursively extracting those files produces 361 `.npy` force trials:

- s001: 54
- s002: 54
- s003: 57
- s004: 45
- s005: 52
- s006: 60
- s007: 39

`bootstrap_data.py` handles this nested archive structure.

### SMPL-H

The supplied SMPL-H archive contains:

- `male/model.npz`
- `female/model.npz`
- `neutral/model.npz`

Each model contains 6,890 template vertices and 52 joints, matching the notebook output.

### Tracker checkpoint

The supplied `t3d_offline.pt` uses the original notebook parameter names:

- `inp`
- `body.*.c`
- `body.*.n`
- `rot`
- `con`

The cleaned tracker module uses descriptive names. `load_tracker_checkpoint` performs a deterministic key remap and then loads with `strict=True`.

## Artifact still needed for a full raw preprocessing rerun

The remaining raw dependency is:

```text
~/scratch/moshpp.zip
```

This archive contains the GroundLink MoSh++ `*_stageii.npz` motion fits. Without it, the raw force files cannot be paired with the motion sequences and `dynamics_trials.pt` cannot be regenerated from scratch.

A copy of the original processed cache would also make exact cross-checking faster, but it is not required once the raw MoSh++ archive is available:

```text
~/scratch/dynamics_project/dynamics_trials.pt
```
