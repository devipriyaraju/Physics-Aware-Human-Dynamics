from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe path in {archive}: {member.filename}")
        zf.extractall(destination)


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:xz") as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe path in {archive}: {member.name}")
        tf.extractall(destination, filter="data")


def extract_nested_zips(root: Path) -> None:
    pending = list(root.rglob("*.zip"))
    seen = set()
    while pending:
        archive = pending.pop(0)
        if archive in seen:
            continue
        seen.add(archive)
        destination = archive.parent / archive.stem
        safe_extract_zip(archive, destination)
        pending.extend(p for p in destination.rglob("*.zip") if p not in seen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract locally obtained GroundLink and SMPL-H archives into the ignored data tree."
    )
    parser.add_argument("--archives", type=Path, default=Path("archives"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    force_zip = args.archives / "force.zip"
    moshpp_zip = args.archives / "moshpp.zip"
    smplh_tar = args.archives / "smplh.tar.xz"

    if force_zip.exists():
        force_root = args.data_root / "groundlink" / "force"
        safe_extract_zip(force_zip, force_root)
        extract_nested_zips(force_root)
        print("force files", len(list(force_root.rglob("*.npy"))))
    else:
        print("missing", force_zip)

    if moshpp_zip.exists():
        motion_root = args.data_root / "groundlink" / "moshpp"
        safe_extract_zip(moshpp_zip, motion_root)
        extract_nested_zips(motion_root)
        print("motion files", len(list(motion_root.rglob("*_stageii.npz"))))
    else:
        print("missing", moshpp_zip)

    if smplh_tar.exists():
        smplh_root = args.data_root / "smplh"
        safe_extract_tar(smplh_tar, smplh_root)
        print("SMPL-H models", len(list(smplh_root.rglob("model.npz"))))
    else:
        print("missing", smplh_tar)


if __name__ == "__main__":
    main()
