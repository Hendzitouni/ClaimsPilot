"""Download the car damage dataset from Roboflow Universe in YOLOv8 format.

Usage (from the repo root, venv activated):
    export ROBOFLOW_API_KEY=...        # never commit this key
    python -m agents.vision.download_data
"""
import os
import sys
from pathlib import Path

import yaml
from roboflow import Roboflow

CONFIG_PATH = Path(os.getenv("VISION_CONFIG", "agents/vision/config.yaml"))


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    ds_cfg = cfg["dataset"]

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY is not set.")

    out_dir = Path(ds_cfg["local_dir"])
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ds_cfg["workspace"]).project(ds_cfg["project"])
    project.version(ds_cfg["version"]).download(ds_cfg["format"], location=str(out_dir))

    # Sanity check: every dataset class must have a Damage mapping.
    names = yaml.safe_load((out_dir / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    mapping = yaml.safe_load(Path(cfg["class_mapping"]).read_text(encoding="utf-8"))
    missing = [n for n in names if n not in mapping]

    print(f"\nDataset classes ({len(names)}): {names}")
    for split in ("train", "valid", "test"):
        imgs = out_dir / split / "images"
        count = len(list(imgs.glob("*"))) if imgs.exists() else 0
        print(f"  {split:5s}: {count} images")
    print("Missing in class_mapping.yaml:", missing or "none")


if __name__ == "__main__":
    main()
