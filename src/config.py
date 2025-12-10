import torch
import os

class Config:
    # Reproducibility
    SEED = 42
    
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = os.cpu_count() or 2
    
    # Hyperparameters
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    
    # Model
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    
    # Paths (Relative to project root)
    DATA_DIR = os.path.join(os.getcwd(), "data")
    MODEL_SAVE_PATH = "best_model.pth"
    
    # ATR Classes Mapping (For reference)
    CLOTHING_IDS = [4, 5, 6, 7, 8, 17]  # Upper, Skirt, Pants, Dress, Belt, Scarf