import torch.nn as nn
import segmentation_models_pytorch as smp

class HybridLoss(nn.Module):
    """Combines Dice Loss (for overlap) and BCE (for pixel accuracy)."""
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.bce(logits, targets)