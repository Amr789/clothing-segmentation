# 👕 Clothing Segmentation Pipeline

## 📄 Overview
This project implements a U-Net based semantic segmentation model to automatically detect and extract clothing items from images. It uses **PyTorch**, **Segmentation Models PyTorch (SMP)**, and **Albumentations** for robust data augmentation.

The model is designed to produce high-quality binary masks (Clothing vs. Background), enabling "Virtual Try-On" applications by creating transparent cutouts of clothing.

## 🚀 Key Features
* **Architecture:** U-Net with MobileNetV2 backbone (Pre-trained on ImageNet).
* **Data Pipeline:** Efficient multi-threaded downloading and processing of the ATR dataset.
* **Loss Function:** Hybrid Loss (Dice Loss + Binary Cross Entropy) for balanced optimization.
* **Inference:** Production-ready inference class that handles resizing, padding, and RGBA cutout generation.

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* GPU recommended (CUDA)

### Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/Amr789/clothing-segmentation](https://github.com/Amr789/clothing-segmentation)
   cd clothing-segmentation
