"""Split the Roboflow export into train/valid/test without leakage.

The Roboflow export contains ~3 augmented copies of each source photo
(`<name>_jpg.rf.<hash>.jpg`), and some source photos were uploaded twice
(`<name> - Copy`). Splitting image by image would put copies of the same car
in train and test, inflating mAP50. We therefore split by *source photo group*.

Usage (from the repo root):
    python -m agents.vision.split_dataset
"""
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CONFIG_PATH = Path(os.getenv("VISION_CONFIG", "agents/vision/config.yaml"))
SPLITS = ("train", "valid", "test")


def group_key(image_name: str) -> str:
    """Map an augmented Roboflow file name to its source-photo group."""
    key = image_name.split(".rf.")[0]                       # drop augmentation hash
    key = re.sub(r"_(jpe?g|png|bmp|webp)$", "", key, flags=re.IGNORECASE)  # drop "_jpg"
    key = re.sub(r"[-_ ]*copy.*$", "", key, flags=re.IGNORECASE)     # "12 - Copy" -> "12"
    return key.strip("-_ ").lower()


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    src = Path(cfg["dataset"]["local_dir"])
    split_cfg = cfg["split"]
    out = Path(split_cfg["output_dir"])
    ratios = split_cfg["ratios"]
    seed = split_cfg["seed"]

    src_yaml = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = src_yaml["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)

    # Gather every image from every split of the export (it may only have "train").
    groups: dict[str, list[Path]] = defaultdict(list)
    for split in SPLITS:
        img_dir = src / split / "images"
        if img_dir.exists():
            for img in sorted(img_dir.iterdir()):
                groups[group_key(img.name)].append(img)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_train = round(len(keys) * ratios["train"])
    n_valid = round(len(keys) * ratios["valid"])
    assignment = {
        "train": keys[:n_train],
        "valid": keys[n_train : n_train + n_valid],
        "test": keys[n_train + n_valid :],
    }

    if out.exists():
        shutil.rmtree(out)
    class_counts = {s: Counter() for s in SPLITS}
    for split, split_keys in assignment.items():
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        for key in split_keys:
            for img in groups[key]:
                label = img.parent.parent / "labels" / f"{img.stem}.txt"
                shutil.copy2(img, out / split / "images" / img.name)
                if label.exists():
                    shutil.copy2(label, out / split / "labels" / label.name)
                    for line in label.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            class_counts[split][names[int(line.split()[0])]] += 1

    # No "path" key: Ultralytics then resolves paths relative to this yaml's folder.
    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print(f"Source photo groups: {len(keys)}  (seed={seed})")
    for split in SPLITS:
        n_img = len(list((out / split / "images").iterdir()))
        print(f"  {split:5s}: {len(assignment[split]):4d} groups, {n_img:5d} images")
    print("\nInstances per class (train / valid / test):")
    for name in names:
        counts = [class_counts[s][name] for s in SPLITS]
        flag = "  <-- missing in valid or test" if 0 in counts[1:] else ""
        print(f"  {name:28s} {counts[0]:5d} {counts[1]:5d} {counts[2]:5d}{flag}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()