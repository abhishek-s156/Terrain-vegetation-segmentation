import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import CLASS_NAMES, normalize_image
from .models import build_model


def colorize(mask):
    palette = np.array([
        [220, 50, 47],    # urban
        [255, 215, 0],    # agriculture
        [181, 101, 29],   # rangeland
        [34, 139, 34],    # forest
        [30, 144, 255],   # water
        [210, 180, 140],  # barren
        [128, 128, 128],  # unknown
    ], dtype=np.uint8)
    return palette[mask]


def predict(image_path, checkpoint_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(ckpt["model"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    tensor = normalize_image(image, ckpt.get("image_size", 256))
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, dict):
            output = output["out"]
        pred = torch.argmax(output, dim=1)[0].cpu().numpy().astype(np.uint8)

    pred_img = Image.fromarray(pred, mode="L").resize(
        original_size, Image.Resampling.NEAREST
    )
    pred_img.save(output_path)

    colored = Image.fromarray(colorize(np.asarray(pred_img)), mode="RGB")
    color_path = Path(output_path).with_name(
        Path(output_path).stem + "_colored.png"
    )
    colored.save(color_path)

    print(f"Class mask: {output_path}")
    print(f"Colored segmentation: {color_path}")
    print("Classes:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output",
        default="outputs/predictions/prediction.png"
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predict(args.image, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
