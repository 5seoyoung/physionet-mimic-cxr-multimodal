import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.t = temperature
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor):
        hard = self.bce(student_logits, targets)
        soft = F.kl_div(
            input=F.logsigmoid(student_logits / self.t),
            target=torch.sigmoid(teacher_logits / self.t),
            reduction="batchmean"
        ) * (self.t ** 2)
        return self.alpha * soft + (1 - self.alpha) * hard
