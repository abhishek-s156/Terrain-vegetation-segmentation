# Terrain & Vegetation Semantic Segmentation

A complete PyTorch project for **semantic segmentation of remote-sensing images** for terrain/vegetation mapping and route/logistics planning.

## What this project does

It supports two segmentation models:

- **U-Net** — implemented from scratch.
- **DeepLabV3-ResNet50** — torchvision implementation.

The target dataset format is compatible with the **DeepGlobe Land Cover** semantic-segmentation dataset.

Classes used by this project:

| ID | Class |
|---:|---|
| 0 | Urban |
| 1 | Agriculture |
| 2 | Rangeland |
| 3 | Forest |
| 4 | Water |
| 5 | Barren |
| 6 | Unknown |

The project includes a **synthetic demo dataset generator**, so the entire pipeline can be tested immediately without downloading a multi-GB satellite dataset.

## Folder structure

```text
terrain_vegetation_segmentation/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── checkpoints/
│   └── predictions/
├── scripts/
│   ├── create_demo_dataset.py
│   └── prepare_deepglobe.py
├── src/
│   ├── dataset.py
│   ├── metrics.py
│   ├── models.py
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## 1. Open in VS Code

Extract the ZIP and open the project folder in VS Code.

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run immediately with the built-in demo data

Generate small synthetic terrain-like RGB images and segmentation masks:

```powershell
python scripts/create_demo_dataset.py
```

Train U-Net:

```powershell
python -m src.train --data_dir data/processed --model unet --epochs 5 --image_size 256
```

For a quick CPU test:

```powershell
python -m src.train --data_dir data/processed --model unet --epochs 1 --image_size 128 --batch_size 2
```

Run prediction:

```powershell
python -m src.predict --image data/processed/val/images/0000.png --checkpoint outputs/checkpoints/best_unet.pt
```

The prediction is saved in:

```text
outputs/predictions/
```

## 5. Launch the visual web app

```powershell
streamlit run app.py
```

Then upload an RGB image and select the trained checkpoint.

## 6. Use a real remote-sensing dataset

This project is prepared for the **DeepGlobe Land Cover** dataset.

DeepGlobe provides RGB satellite images and pixel-level masks for semantic segmentation. The dataset contains classes including urban, agriculture, rangeland, forest, water, barren and unknown.

Download the dataset from the official challenge/Kaggle source, then organize it like this:

```text
data/
└── raw/
    └── deepglobe/
        ├── train/
        │   ├── 001_sat.jpg
        │   ├── 001_mask.png
        │   ├── 002_sat.jpg
        │   ├── 002_mask.png
        │   └── ...
```

Then run:

```powershell
python scripts/prepare_deepglobe.py --input_dir data/raw/deepglobe/train --output_dir data/processed
```

This creates:

```text
data/processed/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

Then train:

```powershell
python -m src.train --data_dir data/processed --model unet --epochs 20 --image_size 256
```

Or use DeepLabV3:

```powershell
python -m src.train --data_dir data/processed --model deeplabv3 --epochs 20 --image_size 256
```

## 7. Important note about real datasets

The full DeepGlobe/LoveDA datasets are large. Do not put them inside the Git repository.

For a student project, start with a subset, for example:

- 100–300 training images
- 20–50 validation images
- image size 256 or 512

Once the pipeline works, increase the dataset and image size.

## 8. Training outputs

The best model is written to:

```text
outputs/checkpoints/best_unet.pt
```

or

```text
outputs/checkpoints/best_deeplabv3.pt
```

Training metrics are printed every epoch:

- loss
- pixel accuracy
- mean IoU

## 9. What the segmentation means for logistics

The predicted map can be used as a preprocessing layer for route planning:

- Roads/urban areas → potentially accessible areas
- Water → high traversal cost / obstacle
- Forest → higher traversal cost
- Agriculture → restricted or medium cost
- Barren/rangeland → potentially lower traversal cost

A future routing module can convert the segmentation map into a **cost map** and run A*, Dijkstra, or another path-planning algorithm.

This project intentionally keeps route planning separate from segmentation so the ML component can be evaluated independently.

## 10. Suggested resume description

> Developed a PyTorch-based semantic segmentation system using U-Net and DeepLabV3 for pixel-level remote-sensing land-cover mapping, supporting terrain/vegetation classification into urban, agriculture, rangeland, forest, water and barren regions, with IoU evaluation and an interactive Streamlit inference interface.

## Dataset references

DeepGlobe Land Cover is a remote-sensing semantic segmentation dataset. The official DeepGlobe resource page explains its dataset/challenge access and usage conditions.

LoveDA is another public remote-sensing land-cover semantic segmentation benchmark with 5,987 high-resolution images and seven labeled land-cover categories. It is a good alternative for future experimentation.

Always check the current dataset license/terms before using imagery commercially.
