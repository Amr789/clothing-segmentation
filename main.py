import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import seed_everything, visualize_batch
from src.data_setup import prepare_dataset
from src.dataset import ClothingDataset, get_transforms
from src.model import build_model
from src.loss import HybridLoss
from src.engine import train_fn, eval_fn
from src.inference import InferencePipeline

def run_training(args):
    # Data
    train_ds = ClothingDataset(
        f"{Config.DATA_DIR}/images/train", 
        f"{Config.DATA_DIR}/masks/train", 
        transform=get_transforms("train")
    )
    val_ds = ClothingDataset(
        f"{Config.DATA_DIR}/images/val", 
        f"{Config.DATA_DIR}/masks/val", 
        transform=get_transforms("val")
    )
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    # Model & Training Components
    model = build_model().to(Config.DEVICE)
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_iou = 0.0
    print(f"🚀 Starting Training for {args.epochs} Epochs...")

    for epoch in range(args.epochs):
        train_loss = train_fn(train_loader, model, optimizer, criterion, Config.DEVICE)
        val_loss, val_iou = eval_fn(val_loader, model, criterion, Config.DEVICE)
        
        scheduler.step(val_loss)

        if val_iou > best_iou:
            print(f"🔥 New Best IoU: {val_iou:.4f}. Saving model...")
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            best_iou = val_iou
            
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

def run_prediction(args):
    pipeline = InferencePipeline(Config.MODEL_SAVE_PATH)
    original, mask, cutout = pipeline.predict(args.image)
    
    # Simple save instead of show for CLI
    cutout.save("output_cutout.png")
    print(f"✅ Prediction saved to output_cutout.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clothing Segmentation CLI")
    parser.add_argument("--mode", type=str, required=True, choices=["setup", "train", "predict"], help="Mode: setup data, train model, or predict")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Number of epochs for training")
    parser.add_argument("--image", type=str, help="Path to image for prediction")
    
    args = parser.parse_args()
    
    seed_everything(Config.SEED)
    
    if args.mode == "setup":
        prepare_dataset()
    elif args.mode == "train":
        run_training(args)
    elif args.mode == "predict":
        if not args.image:
            print("❌ Error: --image path required for prediction mode.")
        else:
            run_prediction(args)