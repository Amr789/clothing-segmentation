import segmentation_models_pytorch as smp
from src.config import Config

def build_model(encoder_name=Config.ENCODER, weights=Config.ENCODER_WEIGHTS):
    print(f"🏗️ Building U-Net with {encoder_name} backbone...")
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        activation=None
    )
    return model