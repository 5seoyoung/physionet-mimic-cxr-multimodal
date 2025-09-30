import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights

class StudentImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 1024, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        self.backbone = m.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(576, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.pool(self.backbone(x)).flatten(1)
        return self.proj(f)
