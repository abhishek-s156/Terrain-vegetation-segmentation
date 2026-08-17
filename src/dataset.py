from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = [
    "urban", "agriculture", "rangeland",
    "forest", "water", "barren", "unknown"
]

# DeepGlobe RGB mask colors used by the original dataset.
# The loader also supports already-indexed grayscale masks.
DEEPGLOBE_COLORS = {
    (0, 255, 255): 0,   # urban
    (255, 255, 0): 1,   # agriculture
    (255, 0, 255): 2,   # rangeland
    (0, 255, 0): 3,     # forest
    (0, 0, 255): 4,     # water
    (255, 255, 255): 5, # barren
    (0, 0, 0): 6,      # unknown
}


def mask_to_class(mask: Image.Image) -> np.ndarray:
    arr = np.array(mask)

    if arr.ndim == 2:
        return arr.astype(np.int64)

    out = np.full(arr.shape[:2], 6, dtype=np.int64)
    for rgb, idx in DEEPGLOBE_COLORS.items():
        out[np.all(arr[:, :, :3] == np.array(rgb), axis=2)] = idx
    return out


def normalize_image(image: Image.Image, size: int):
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    # ImageNet normalization works well for torchvision backbones.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


class LandCoverDataset(Dataset):
    def __init__(self, root, split="train", image_size=256, augment=False):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Expected {self.image_dir} and {self.mask_dir}. "
                "Run create_demo_dataset.py or prepare_deepglobe.py first."
            )

        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in image_exts
        )

        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def _mask_path(self, image_path):
        candidates = [
            self.mask_dir / image_path.name,
            self.mask_dir / f"{image_path.stem}.png",
            self.mask_dir / f"{image_path.stem}.jpg",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"No mask found for {image_path.name}")

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self._mask_path(image_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # Same random horizontal flip for image and mask.
        if self.augment and np.random.rand() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if self.augment and np.random.rand() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        image_tensor = normalize_image(image, self.image_size)

        mask = mask_to_class(mask)
        mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
        mask_img = mask_img.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )
        mask_tensor = torch.from_numpy(np.asarray(mask_img).astype(np.int64))

        return image_tensor, mask_tensor
