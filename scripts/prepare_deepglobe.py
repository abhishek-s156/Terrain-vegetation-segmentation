import argparse
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

IMAGE_SUFFIXES = ["_sat.jpg", "_sat.jpeg", "_sat.png"]


def find_images(folder):
    images = []
    for suffix in IMAGE_SUFFIXES:
        images.extend(Path(folder).glob(f"*{suffix}"))
    return sorted(set(images))


def find_mask(image_path):
    stem = image_path.name
    if "_sat" in stem:
        base = stem.split("_sat")[0]
    else:
        base = image_path.stem

    candidates = [
        image_path.parent / f"{base}_mask.png",
        image_path.parent / f"{base}_mask.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--val_size", type=float, default=0.15)
    args = parser.parse_args()

    images = find_images(args.input_dir)
    pairs = [(p, find_mask(p)) for p in images]
    pairs = [(x, y) for x, y in pairs if y is not None]

    if not pairs:
        raise RuntimeError(
            "No image/mask pairs found. Expected files like 123_sat.jpg and 123_mask.png."
        )

    train_pairs, val_pairs = train_test_split(
        pairs, test_size=args.val_size, random_state=42
    )

    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out = Path(args.output_dir) / split / "images"
        mask_out = Path(args.output_dir) / split / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for image, mask in split_pairs:
            shutil.copy2(image, img_out / image.name)
            shutil.copy2(mask, mask_out / mask.name)

    print(f"Found {len(pairs)} labeled pairs.")
    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"Prepared dataset at {args.output_dir}")


if __name__ == "__main__":
    main()
