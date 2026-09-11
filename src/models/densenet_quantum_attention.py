"""
Phase 2: DenseNet121 backbone + QuantumChannelAttention head.

This mirrors DenseNetASL (densenet_baseline.py) but inserts a quantum
channel-attention block between the pooled feature vector and the final
classifier. Compare this model's val/test accuracy and parameter count
directly against:
  - DenseNetASL (no attention)               -> Phase 0
  - DenseNetASL + a classical SE/CBAM block   -> Phase 0 classical-attention baseline

Recommended first run: --freeze-backbone, so only the quantum attention
block + classifier head are trained. This is cheap and is the right way to
first check "does the quantum attention layer learn anything useful on top
of frozen DenseNet features" before paying for full fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

from .quantum_attention import QuantumChannelAttention


class DenseNetQuantumAttention(nn.Module):
    def __init__(self, num_classes: int, n_qubits: int = 8, n_layers: int = 2,
                 pretrained: bool = True, freeze_backbone: bool = True,
                 device_name: str = "default.qubit",
                 quantum_architecture: str = "v1"):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        self.features = backbone.features
        feature_dim = backbone.classifier.in_features  # 1024

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.q_attn = QuantumChannelAttention(
            feature_dim=feature_dim, n_qubits=n_qubits, n_layers=n_layers,
            device_name=device_name, architecture=quantum_architecture,
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = torch.relu(f)
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        f = self.forward_features(x)
        f_attn, attn = self.q_attn(f)
        logits = self.classifier(f_attn)
        if return_attn:
            return logits, attn
        return logits


def build_model(num_classes: int, n_qubits: int = 8, n_layers: int = 2,
                 pretrained: bool = True, freeze_backbone: bool = True,
                 device_name: str = "default.qubit",
                 quantum_architecture: str = "v1") -> DenseNetQuantumAttention:
    return DenseNetQuantumAttention(
        num_classes=num_classes, n_qubits=n_qubits, n_layers=n_layers,
        pretrained=pretrained, freeze_backbone=freeze_backbone,
        device_name=device_name, quantum_architecture=quantum_architecture,
    )


if __name__ == "__main__":
    # Smoke test (downloads ImageNet weights on first run)
    m = build_model(num_classes=29, pretrained=False, freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    logits, attn = m(x, return_attn=True)
    print("logits shape:", logits.shape)  # (2, 29)
    print("attn shape:  ", attn.shape)    # (2, 1024)

    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {n_trainable:,} / total: {n_total:,}")
