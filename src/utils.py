import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def seed_everything(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold=0.5) -> float:
    """Calculates Intersection over Union (IoU) for binary segmentation."""
    pred_mask = (pred_mask > threshold).float()
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    
    if union == 0:
        return 1.0
    return (intersection / union).item()

def visualize_batch(images, masks, preds=None, save_path=None):
    """Visualizes a batch of images and masks."""
    batch_size = len(images)
    cols = 3 if preds is not None else 2
    fig, axes = plt.subplots(batch_size, cols, figsize=(10, 4 * batch_size))
    
    if batch_size == 1: axes = [axes] # Handle single image case

    for i in range(batch_size):
        # Un-normalize image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Plot Original
        ax_img = axes[i][0] if batch_size > 1 else axes[0]
        ax_img.imshow(img)
        ax_img.set_title("Input Image")
        ax_img.axis('off')

        # Plot Ground Truth
        ax_mask = axes[i][1] if batch_size > 1 else axes[1]
        ax_mask.imshow(masks[i].cpu().numpy(), cmap='gray')
        ax_mask.set_title("Ground Truth")
        ax_mask.axis('off')
        
        # Plot Prediction (if available)
        if preds is not None:
            ax_pred = axes[i][2] if batch_size > 1 else axes[2]
            ax_pred.imshow(preds[i].cpu().squeeze().numpy(), cmap='jet', alpha=0.8)
            ax_pred.set_title("Prediction")
            ax_pred.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Visualization saved to {save_path}")
    else:
        plt.show()