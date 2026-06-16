"""
Quantum attention modules (Phase 1+).

Two variants, per the research plan:

1. QuantumChannelAttention (implemented, testable now)
   - Operates on the pooled 1024-d DenseNet feature vector.
   - Classically project -> n_qubits, angle-encode, run a variational
     circuit, measure <Z> on each qubit, project back up to feature_dim,
     sigmoid -> per-channel attention weights that reweight the feature
     vector (SE-block-style, but the gating function is quantum).
   - Inspired by the QCNN channel-attention idea (arXiv:2311.02871).

2. QuantumTokenSelfAttention (stub, for Phase 3)
   - QSANN-style: treat DenseNet's spatial feature map (C, H, W) as a
     sequence of H*W tokens, run separate small variational circuits as
     quantum Q/K/V, compute attention via Gaussian-projected measurements
     (arXiv:2205.05625) or APDM (Applied Intelligence 2024).
   - Left as a stub -- get QuantumChannelAttention validated first, since
     it's far cheaper (one circuit per sample vs. one per token).

Run this file directly for a smoke test that only needs PennyLane + torch
(no DenseNet / dataset required):

    python -m models.quantum_attention
"""

import math

import pennylane as qml
import torch
import torch.nn as nn


class QuantumChannelAttention(nn.Module):
    """
    Quantum channel-attention block.

    Args:
        feature_dim: dimensionality of the input feature vector (e.g. 1024
            for DenseNet121's pooled output).
        n_qubits: number of qubits used for the variational circuit. Keep
            this small (4-10) -- simulation cost scales as 2^n_qubits.
        n_layers: number of StronglyEntanglingLayers in the PQC.
        device_name: PennyLane device. "default.qubit" supports
            diff_method="backprop" (fast for prototyping). Switch to
            "lightning.qubit" + diff_method="adjoint" for larger n_qubits.
    """

    def __init__(self, feature_dim: int, n_qubits: int = 8, n_layers: int = 2,
                 device_name: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits

        dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Angle embedding: classical features -> rotation angles
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            # Entangling variational layers (this is the "trainable" part)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # Measurement: one expectation value per qubit, in [-1, 1]
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {
            "weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        }
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Classical down/up projections around the quantum core
        self.down_proj = nn.Linear(feature_dim, n_qubits)
        self.up_proj = nn.Linear(n_qubits, feature_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, feature_dim)
        Returns:
            (attended_features, attention_weights), both (B, feature_dim)
        """
        z = self.down_proj(x)
        # AngleEmbedding expects angles; squash to (-pi/2, pi/2) for a
        # well-behaved encoding range.
        z = torch.tanh(z) * (math.pi / 2)

        # PennyLane's default.qubit simulator always runs on CPU.
        # Bridge the device gap: move to CPU for the quantum circuit,
        # then return to the original device (e.g. MPS, CUDA) afterwards.
        src_device = z.device
        q_out = self.qlayer(z.cpu()).to(src_device)  # (B, n_qubits)

        attn = torch.sigmoid(self.up_proj(q_out))  # (B, feature_dim)
        return x * attn, attn


class QuantumTokenSelfAttention(nn.Module):
    """
    STUB -- Phase 3.

    QSANN-style quantum self-attention over spatial tokens from a DenseNet
    feature map. Sketch of the intended design:

      1. Input: feature map (B, C, H, W) -> reshape to (B, H*W, C) tokens.
      2. Linear-project each token from C -> n_qubits.
      3. For each token, encode into an n_qubit state (AngleEmbedding).
      4. Apply three small PQCs (or one PQC + APDM) to obtain per-token
         quantum query/key/value vectors.
      5. Compute attention scores classically (e.g. scaled dot product, or
         Gaussian-projected as in QSANN) and aggregate values.
      6. Project the aggregated, attended tokens back to C and reshape to
         (B, C, H, W) (or pool to (B, C)).

    Cost warning: this runs one circuit per token, so for a 7x7 DenseNet121
    feature map that's 49 circuit evaluations per sample per forward pass.
    Validate QuantumChannelAttention end-to-end first; if that pipeline
    works, downsample the feature map (e.g. to 2x2 or 3x3 tokens via extra
    pooling) before attempting this.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "QuantumTokenSelfAttention is a design stub -- implement after "
            "QuantumChannelAttention is validated (see docstring)."
        )


if __name__ == "__main__":
    # Smoke test: does PennyLane + the TorchLayer work end-to-end and is it
    # differentiable? No DenseNet or dataset needed.
    torch.manual_seed(0)

    feature_dim = 1024
    block = QuantumChannelAttention(feature_dim=feature_dim, n_qubits=6, n_layers=2)

    x = torch.randn(4, feature_dim, requires_grad=True)
    out, attn = block(x)

    print("input shape: ", x.shape)
    print("output shape:", out.shape)
    print("attn shape:  ", attn.shape)
    print("attn range:  [{:.4f}, {:.4f}]".format(attn.min().item(), attn.max().item()))

    loss = out.sum()
    loss.backward()
    print("grad on input is None?", x.grad is None)
    n_q_params = sum(p.numel() for p in block.qlayer.parameters())
    print(f"quantum layer trainable params: {n_q_params}")
