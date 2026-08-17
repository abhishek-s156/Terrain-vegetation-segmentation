from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)
np.random.seed(42)

OUT = Path("data/processed")
SIZE = 256
CLASSES = 7

# Simple visual palette for synthetic satellite-like terrain.
RGB = np.array([
    [155, 80, 65],     # urban
    [205, 190, 80],    # agriculture
    [180, 135, 80],    # rangeland
    [50, 125, 55],     # forest
    [35, 100, 190],    # water
    [190, 170, 145],   # barren
    [125, 125, 125],   # unknown
], dtype=np.uint8)


def make_one(seed):
    rng = np.random.default_rng(seed)
    mask = np.full((SIZE, SIZE), 5, dtype=np.uint8)

    # Large terrain regions.
    for _ in range(18):
        cls = int(rng.integers(0, 6))
        x = int(rng.integers(0, SIZE))
        y = int(rng.integers(0, SIZE))
        w = int(rng.integers(25, 100))
        h = int(rng.integers(20, 100))

        yy, xx = np.ogrid[:SIZE, :SIZE]
        ellipse = ((xx - x) / max(w, 1)) ** 2 + ((yy - y) / max(h, 1)) ** 2 < 1
        mask[ellipse] = cls

    # Make a winding river.
    yy = np.arange(SIZE)
    center = SIZE * 0.48 + 35 * np.sin(yy / 35.0)
    for y in range(SIZE):
        c = int(center[y])
        left = max(0, c - 7)
        right = min(SIZE, c + 8)
        mask[y, left:right] = 4

    # Urban blocks.
    for _ in range(18):
        x = int(rng.integers(0, SIZE - 20))
        y = int(rng.integers(0, SIZE - 20))
        w = int(rng.integers(6, 22))
        h = int(rng.integers(6, 22))
        mask[y:y+h, x:x+w] = 0

    # Add a few roads as urban pixels.
    for y in range(20, SIZE, 50):
        mask[max(0, y-2):min(SIZE, y+3), :] = 0

    # Image = class color + texture/noise.
    image = RGB[mask].astype(np.int16)
    noise = rng.normal(0, 12, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # Slight blur/noise to look more like remote-sensing imagery.
    pil = Image.fromarray(image, "RGB").filter(ImageFilter.GaussianBlur(0.6))
    return pil, Image.fromarray(mask, "L")


def make_split(split, count, start):
    img_dir = OUT / split / "images"
    mask_dir = OUT / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        image, mask = make_one(start + i)
        name = f"{i:04d}"
        image.save(img_dir / f"{name}.png")
        mask.save(mask_dir / f"{name}.png")


if __name__ == "__main__":
    make_split("train", 40, 100)
    make_split("val", 10, 1000)
    print("Created demo dataset:")
    print("  data/processed/train/images")
    print("  data/processed/train/masks")
    print("  data/processed/val/images")
    print("  data/processed/val/masks")
