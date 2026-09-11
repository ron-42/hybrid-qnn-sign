"""
Phase 0b: classical channel-attention baselines for parameter-efficiency comparison.

Both SE and CBAM plug in at the same point as QuantumChannelAttention:
the pooled 1024-d feature vector from DenseNet121. This makes parameter
counts directly comparable across all three attention variants.

Run this file directly for a smoke test:
    python -m src.models.classical_attention
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


# ── SE block ──────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).

    Operates on a pooled feature vector (B, feature_dim) — same contract as
    QuantumChannelAttention so parameter counts are directly comparable.
    """

    def __init__(self, feature_dim: int, reduction: int = 16):
        super().__init__()
        hidden = max(feature_dim // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, feature_dim)
        Returns:
            (attended_features, attention_weights) both (B, feature_dim)
        """
        attn = self.fc(x)
        return x * attn, attn


# ── Parameter-matched MLP block ───────────────────────────────────────────────

class MLPChannelAttention(nn.Module):
    """Classical replacement for QCA with the same down/up projection sizes."""

    def __init__(self, feature_dim: int, bottleneck_dim: int = 8):
        super().__init__()
        self.down_proj = nn.Linear(feature_dim, bottleneck_dim)
        self.core = nn.Linear(bottleneck_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, feature_dim)

    def forward(self, x: torch.Tensor):
        z = torch.tanh(self.down_proj(x))
        z = torch.tanh(self.core(z))
        attn = torch.sigmoid(self.up_proj(z))
        return x * attn, attn


class DenseNetMLPAttention(nn.Module):
    """DenseNet121 + parameter-matched classical MLP channel attention."""

    def __init__(self, num_classes: int, bottleneck_dim: int = 8,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        self.features = backbone.features
        feature_dim = backbone.classifier.in_features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_attn = MLPChannelAttention(feature_dim, bottleneck_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = torch.relu(self.features(x))
        return self.pool(f).flatten(1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        f_attn, attn = self.mlp_attn(self.forward_features(x))
        logits = self.classifier(f_attn)
        if return_attn:
            return logits, attn
        return logits


def build_mlp_model(num_classes: int, bottleneck_dim: int = 8,
                    pretrained: bool = True,
                    freeze_backbone: bool = False) -> DenseNetMLPAttention:
    return DenseNetMLPAttention(
        num_classes=num_classes,
        bottleneck_dim=bottleneck_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


# ── CBAM channel sub-block (operates on pooled vector) ────────────────────────

class CBAMChannelBlock(nn.Module):
    """
    CBAM channel-attention sub-block (Woo et al., ECCV 2018), adapted for
    a pre-pooled feature vector.  Replaces the spatial attention portion since
    we're operating on a 1-D feature vector (spatial info already collapsed).
    """

    def __init__(self, feature_dim: int, reduction: int = 16):
        super().__init__()
        hidden = max(feature_dim // reduction, 4)
        self.shared_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim, bias=False),
        )

    def forward(self, x: torch.Tensor):
        # CBAM uses both avg-pool and max-pool channel descriptors
        avg_out = self.shared_mlp(x)
        max_out = self.shared_mlp(x)   # x is already avg-pooled; max path mirrors the original
        attn = torch.sigmoid(avg_out + max_out)
        return x * attn, attn


# ── DenseNet + SE ─────────────────────────────────────────────────────────────

class DenseNetSEAttention(nn.Module):
    """DenseNet121 + SE channel-attention block (Phase 0b comparison model)."""

    def __init__(self, num_classes: int, reduction: int = 16,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        self.features = backbone.features
        feature_dim = backbone.classifier.in_features  # 1024

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = SEBlock(feature_dim, reduction=reduction)
        self.classifier = nn.Linear(feature_dim, num_classes)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = torch.relu(f)
        return self.pool(f).flatten(1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        f = self.forward_features(x)
        f_attn, attn = self.se(f)
        logits = self.classifier(f_attn)
        if return_attn:
            return logits, attn
        return logits


def build_model(num_classes: int, reduction: int = 16,
                pretrained: bool = True, freeze_backbone: bool = False) -> DenseNetSEAttention:
    return DenseNetSEAttention(
        num_classes=num_classes, reduction=reduction,
        pretrained=pretrained, freeze_backbone=freeze_backbone,
    )


# ── DenseNet + CBAM ───────────────────────────────────────────────────────────

class DenseNetCBAMAttention(nn.Module):
    """DenseNet121 + CBAM channel-attention block."""

    def __init__(self, num_classes: int, reduction: int = 16,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        self.features = backbone.features
        feature_dim = backbone.classifier.in_features  # 1024

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cbam = CBAMChannelBlock(feature_dim, reduction=reduction)
        self.classifier = nn.Linear(feature_dim, num_classes)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = torch.relu(f)
        return self.pool(f).flatten(1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        f = self.forward_features(x)
        f_attn, attn = self.cbam(f)
        logits = self.classifier(f_attn)
        if return_attn:
            return logits, attn
        return logits


def build_cbam_model(num_classes: int, reduction: int = 16,
                     pretrained: bool = True, freeze_backbone: bool = False) -> DenseNetCBAMAttention:
    return DenseNetCBAMAttention(
        num_classes=num_classes, reduction=reduction,
        pretrained=pretrained, freeze_backbone=freeze_backbone,
    )


if __name__ == "__main__":
    models = [
        (DenseNetSEAttention, "SE", "se"),
        (DenseNetCBAMAttention, "CBAM", "cbam"),
        (DenseNetMLPAttention, "MLP", "mlp_attn"),
    ]
    for ModelCls, name, attention_attr in models:
        m = ModelCls(num_classes=29, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        logits, attn = m(x, return_attn=True)
        n_attn = sum(p.numel() for p in getattr(m, attention_attr).parameters())
        n_total = sum(p.numel() for p in m.parameters())
        print(f"[{name}] logits={logits.shape}  attn={attn.shape}  "
              f"attn_params={n_attn:,}  total_params={n_total:,}")
