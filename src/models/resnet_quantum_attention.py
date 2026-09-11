"""
ResNet50 backbone + QuantumChannelAttention head for ASL Alphabet.

The quantum channel-attention block is applied after ResNet's global average
pooling and before its classifier, matching the DenseNet quantum experiment.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from .quantum_attention import QuantumChannelAttention


class ResNetQuantumAttention(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        device_name: str = "default.qubit",
        quantum_architecture: str = "v1",
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = backbone.avgpool
        feature_dim = backbone.fc.in_features  # 2048
        self.q_attn = QuantumChannelAttention(
            feature_dim=feature_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=device_name,
            architecture=quantum_architecture,
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.backbone(x)).flatten(1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        features = self.forward_features(x)
        attended_features, attn = self.q_attn(features)
        logits = self.classifier(attended_features)
        if return_attn:
            return logits, attn
        return logits


def build_model(
    num_classes: int,
    n_qubits: int = 8,
    n_layers: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device_name: str = "default.qubit",
    quantum_architecture: str = "v1",
) -> ResNetQuantumAttention:
    return ResNetQuantumAttention(
        num_classes=num_classes,
        n_qubits=n_qubits,
        n_layers=n_layers,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        device_name=device_name,
        quantum_architecture=quantum_architecture,
    )


if __name__ == "__main__":
    model = build_model(num_classes=29, pretrained=False, freeze_backbone=True)
    inputs = torch.randn(2, 3, 224, 224)
    logits, attn = model(inputs, return_attn=True)
    assert logits.shape == (2, 29)
    assert attn.shape == (2, 2048)
    print("logits shape:", logits.shape)
    print("attn shape:  ", attn.shape)
    n_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    n_total = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable params: {n_trainable:,} / total: {n_total:,}")
