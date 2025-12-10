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

## ☁️ Run on Google Colab

You can run this project entirely in the cloud using Google Colab's free GPUs (T4).

### 1. Setup Environment
Open a new Colab notebook and run the following in the first cell to clone the repo and install dependencies:

```python
# 1. Clone repository
!git clone https://github.com/Amr789/clothing-segmentation.git

# 2. Enter the directory
%cd clothing-segmentation

# 3. Install dependencies from requirements file
!pip install -r requirements.txt -q

# 4. Mount Google Drive (To save your trained model later)
from google.colab import drive
drive.mount('/content/drive')
