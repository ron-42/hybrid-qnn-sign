"""
DenseNet121 baseline (Phase 0) for ASL Alphabet classification.

Also exposes a `forward_features` method that returns the pooled
penultimate feature vector (1024-d) -- this is the hook point Phase 2
(quantum attention) will plug into later.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNetASL(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        self.features = backbone.features  # conv layers up to final BN, output: (B, 1024, H, W)
        in_features = backbone.classifier.in_features  # 1024 for densenet121

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled 1024-d feature vector (post global-avg-pool, pre-classifier)."""
        f = self.features(x)
        f = torch.relu(f)  # DenseNet applies a final ReLU before pooling
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.forward_features(x)
        return self.classifier(f)


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> DenseNetASL:
    return DenseNetASL(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)


if __name__ == "__main__":
    # Quick smoke test
    m = build_model(num_classes=29, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    feats = m.forward_features(x)
    logits = m(x)
    print("feature shape:", feats.shape)   # expect (2, 1024)
    print("logits shape:", logits.shape)   # expect (2, 29)
