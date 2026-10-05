# Raw data layout

Raw third-party assets are not committed to this repository.

After running `python scripts/bootstrap_data.py`, the expected layout is:

```text
data/raw/
|-- groundlink/
|   |-- force/
|   |   `-- ... .npy force trials
|   `-- moshpp/
|       `-- ... _stageii.npz motion fits
`-- smplh/
    |-- female/model.npz
    |-- male/model.npz
    `-- neutral/model.npz
```

The preprocessing code searches recursively, so intermediate subject directories are allowed.

Do not commit these assets unless their third-party licenses explicitly allow redistribution.
