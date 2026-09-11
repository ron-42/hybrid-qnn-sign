from .densenet_baseline import DenseNetASL, build_model as build_baseline
from .densenet_quantum_attention import DenseNetQuantumAttention, build_model as build_quantum
from .resnet_quantum_attention import ResNetQuantumAttention, build_model as build_resnet_quantum
from .quantum_attention import QuantumChannelAttention, QuantumTokenSelfAttention
from .classical_attention import (
    SEBlock, DenseNetSEAttention, build_model as build_se,
    CBAMChannelBlock, DenseNetCBAMAttention, build_cbam_model,
    MLPChannelAttention, DenseNetMLPAttention, build_mlp_model,
)

__all__ = [
    # Phase 0
    "DenseNetASL", "build_baseline",
    # Phase 0b
    "SEBlock", "DenseNetSEAttention", "build_se",
    "CBAMChannelBlock", "DenseNetCBAMAttention", "build_cbam_model",
    "MLPChannelAttention", "DenseNetMLPAttention", "build_mlp_model",
    # Phase 1 / 2
    "QuantumChannelAttention", "QuantumTokenSelfAttention",
    "DenseNetQuantumAttention", "build_quantum",
    "ResNetQuantumAttention", "build_resnet_quantum",
]
