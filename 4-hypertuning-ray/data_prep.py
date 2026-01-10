"""Download and preprocess the Oxford Flowers (102) dataset.

This script downloads the Flowers102 dataset (via torchvision), resizes images,
and writes them to an output folder with the structure:

  output_dir/{train,val,test}/{class_label}/image.jpg

Usage examples:
  python data_prep.py --dataset flowers --out data/flowers --image-size 128

If `torchvision.datasets.Flowers102` is not available, the script will raise
an informative error and suggest alternatives.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

from PIL import Image
from tqdm import tqdm

import torchvision.transforms as T


def prepare_flowers(root: Path, out: Path, image_size: int, val_split: float, test_split: float, seed: int, clear_out: bool) -> None:
    try:
        from torchvision.datasets import Flowers102
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Could not import Flowers102 from torchvision. "
            "Ensure you have a recent torchvision installed, or download the dataset manually."
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    out = out.expanduser()
    if clear_out and out.exists():
        # don't import shutil at top-level to keep imports minimal
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Flowers102 into {root} (if not already present)...")
    ds = Flowers102(root=str(root), download=True)

    n = len(ds)
    if n == 0:
        raise RuntimeError("Flowers102 dataset contains no samples after download.")

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    transform = T.Compose([T.Resize((image_size, image_size))])

    print(f"Preparing images ({image_size}x{image_size}) into {out}")
    for split, ids in splits.items():
        for i in tqdm(ids, desc=split):
            item = ds[i]
            # Flowers102 returns (image, label) in most torchvision versions
            if isinstance(item, tuple) and len(item) >= 2:
                img, label = item[0], item[1]
            else:
                raise RuntimeError("Unexpected item format from Flowers102 dataset")

            # normalize label to a string folder name
            try:
                label_id = int(label)
            except Exception:
                label_id = int(label.item()) if hasattr(label, "item") else 0

            class_dir = out / split / f"class_{label_id:03d}"
            class_dir.mkdir(parents=True, exist_ok=True)

            img = transform(img)
            if not isinstance(img, Image.Image):
                # transform may return a tensor in some setups, convert back
                from torchvision.transforms.functional import to_pil_image

                img = to_pil_image(img)

            dest = class_dir / f"{i:06d}.jpg"
            img.save(dest)

    print("Done. Prepared dataset layout:")
    for p in sorted(out.iterdir()):
        if p.is_dir():
            print(f" - {p}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Flowers dataset for hypertuning experiments")
    p.add_argument("--dataset", choices=["flowers"], default="flowers", help="Which dataset to prepare")
    p.add_argument("--root", type=Path, default=Path("./data_raw"), help="Raw dataset download folder")
    p.add_argument("--out", type=Path, default=Path("./data/flowers"), help="Output prepared dataset folder")
    p.add_argument("--image-size", type=int, default=128, help="Size to resize images to (square)")
    p.add_argument("--val-split", type=float, default=0.15, help="Fraction for validation set")
    p.add_argument("--test-split", type=float, default=0.15, help="Fraction for test set")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    p.add_argument("--clear-out", action="store_true", help="Clear output folder before writing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset != "flowers":
        raise NotImplementedError("This script currently only supports the 'flowers' dataset")

    prepare_flowers(args.root, args.out, args.image_size, args.val_split, args.test_split, args.seed, args.clear_out)


if __name__ == "__main__":
    main()
