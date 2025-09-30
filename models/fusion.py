import torch
import torch.nn as nn

class LateFusionHead(nn.Module):
    def __init__(self, img_dim: int, txt_dim: int, num_labels: int = 14):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + txt_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels),
        )
    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img_feat, txt_feat], dim=1)
        return self.classifier(x)
