import torch
from tqdm import tqdm
from src.config import Config
from src.utils import calculate_iou

def train_fn(loader, model, optimizer, criterion, device):
    model.train()
    loop = tqdm(loader, desc="Training", leave=False)
    total_loss = 0
    
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def eval_fn(loader, model, criterion, device):
    model.eval()
    loop = tqdm(loader, desc="Validating", leave=False)
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()

            logits = model(images)
            loss = criterion(logits, masks)
            
            probs = torch.sigmoid(logits)
            iou = calculate_iou(probs, masks)

            total_loss += loss.item()
            total_iou += iou

    return (total_loss / len(loader)), (total_iou / len(loader))