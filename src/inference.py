import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from src.config import Config
from src.model import build_model

class InferencePipeline:
    def __init__(self, model_path, device=Config.DEVICE):
        self.device = device
        print(f"⚙️ Loading model weights from {model_path}...")
        
        self.model = build_model() # Uses Config defaults automatically
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def predict(self, input_data):
        if isinstance(input_data, str):
            image = cv2.imread(input_data)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(input_data, Image.Image):
            image = np.array(input_data)
        else:
            image = input_data
            
        original_h, original_w = image.shape[:2]
        
        # Preprocess
        augmented = self.transform(image=image)["image"]
        tensor_img = augmented.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = torch.sigmoid(logits)
            pred_mask_small = (probs > 0.5).float().squeeze().cpu().numpy()

        # Resize back to original
        full_size_mask = cv2.resize(
            pred_mask_small, 
            (original_w, original_h), 
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

        # Create Cutout
        rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = full_size_mask * 255
        clothing_cutout = Image.fromarray(rgba)

        return image, full_size_mask, clothing_cutout