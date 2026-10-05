# Local archives

This directory is intentionally excluded from version control except for this file.

For a full raw-data reproduction, place locally obtained copies of these archives here:

- `force.zip`
- `moshpp.zip`
- `smplh.tar.xz`

Then run:

```bash
python scripts/bootstrap_data.py
python scripts/prepare_data.py
```

`bootstrap_data.py` also handles the GroundLink packaging where `force.zip` contains one ZIP file per subject.

Do not commit GroundLink or SMPL-H assets unless their licenses explicitly permit redistribution.
