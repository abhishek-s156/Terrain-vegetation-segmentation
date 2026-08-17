python scripts/create_demo_dataset.py
python -m src.train --data_dir data/processed --model unet --epochs 1 --image_size 128 --batch_size 2
python -m src.predict --image data/processed/val/images/0000.png --checkpoint outputs/checkpoints/best_unet.pt
streamlit run app.py
