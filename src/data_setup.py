import os
import functools
import numpy as np
import cv2
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from src.config import Config

def save_single_image(item, idx, img_dir, mask_dir):
    try:
        # Save Image
        img_path = os.path.join(img_dir, f"{idx}.jpg")
        item['image'].convert("RGB").save(img_path)
        
        # Save Mask
        mask_path = os.path.join(mask_dir, f"{idx}.png")
        mask = np.array(item['mask'])
        cv2.imwrite(mask_path, mask)
    except Exception as e:
        print(f"⚠️ Error saving image {idx}: {e}")

def prepare_dataset(num_train=400, num_val=50):
    """Downloads and extracts the ATR dataset."""
    train_dir = os.path.join(Config.DATA_DIR, "images/train")
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) >= num_train:
        print("✅ Data already present. Skipping download.")
        return

    print("⬇️ Downloading ATR Dataset (Metadata only)...")
    hf_dataset = load_dataset("mattmdjaga/human_parsing_dataset")

    base = Config.DATA_DIR
    splits = [
        ("train", "train", num_train),
        ("train", "val", num_val) # Using 'train' split for both but splitting manually
    ]

    for hf_split, local_split, count in splits:
        print(f"⚡ Processing {local_split} split ({count} images)...")
        img_dir = os.path.join(base, "images", local_split)
        mask_dir = os.path.join(base, "masks", local_split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        subset = hf_dataset[hf_split].select(range(count))
        worker = functools.partial(save_single_image, img_dir=img_dir, mask_dir=mask_dir)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(tqdm(executor.map(worker, subset, range(count)), total=count))

    print("✅ Data preparation complete!")