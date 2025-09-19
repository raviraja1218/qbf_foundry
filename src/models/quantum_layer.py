# src/models/quantum_layer.py
"""
QuantumLayer: a small PyTorch module that wraps a PennyLane quantum circuit.
- Maps a classical latent vector -> (linear) angles -> quantum circuit -> expectation values
- Maps expectation values back to the same latent dimension with a final linear layer.
This implementation:
 - Accepts batched torch tensors
 - Uses a small, configurable number of qubits and layers
 - Uses TorchLayer(qnode, weight_shapes) without unsupported kwargs
 - Handles dtype/device correctly (PennyLane qnode configured with interface='torch')
"""

from typing import Optional

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.qnn import TorchLayer

class QuantumLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        n_layers: int = 1,
        dev_name: str = "default.qubit",
        wires_offset: int = 0,
        seed: Optional[int] = None,
    ):
        """
        input_dim: size of incoming latent vector
        n_qubits: number of qubits to use in the circuit (<= input_dim ideally)
        n_layers: number of strongly entangling layers
        dev_name: PennyLane device string
        """
        super().__init__()

        if seed is not None:
            qml.devices.default.qubit.seed = seed

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.wires = list(range(wires_offset, wires_offset + n_qubits))

        # Linear to map input_dim -> n_qubits (angles for AngleEmbedding)
        self.pre_angle = nn.Linear(input_dim, n_qubits)

        # Define Pennylane device
        self.dev = qml.device(dev_name, wires=self.wires)

        # shape for variational weights used by StronglyEntanglingLayers:
        # (n_layers, n_wires, 3)
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # Create a QNode that accepts (inputs, weights) named arguments
        def qnode_fn(inputs, weights):
            """
            inputs: 1-D array (n_qubits,) of angles
            weights: shape (n_layers, n_qubits, 3)
            Returns expectation values [Z_0, Z_1, ..., Z_{n_qubits-1}]
            """
            # AngleEmbedding expects an array-like of length n_qubits
            qml.templates.AngleEmbedding(inputs, wires=self.wires, rotation="Y")
            # Variational layers
            qml.templates.StronglyEntanglingLayers(weights, wires=self.wires)
            # Return Pauli-Z expectation on each wire
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        # Wrap qnode with qml.qnode to use 'torch' interface and backprop
        qnode = qml.QNode(qnode_fn, self.dev, interface="torch", diff_method="backprop")

        # TorchLayer expects (qnode, weight_shapes) where qnode signature includes inputs and weights
        self.qlayer = TorchLayer(qnode, self.weight_shapes)

        # Map qout (n_qubits) -> input_dim to get transformed latent vector
        self.post_map = nn.Linear(n_qubits, input_dim)

        # initialize small weight scale for stable outputs
        nn.init.xavier_uniform_(self.pre_angle.weight)
        nn.init.xavier_uniform_(self.post_map.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [B, input_dim] (batch of latent vectors)
        returns: transformed latent [B, input_dim]
        """
        if x.dim() != 2:
            raise ValueError("QuantumLayer expects input with shape [B, input_dim]")

        # Map to angles: [B, n_qubits]
        angles = self.pre_angle(x)  # torch tensor on correct device/dtype

        # PennyLane/TorchLayer accepts batched inputs of shape (B, n_input)
        # Our qnode signature is (inputs, weights) so TorchLayer call expects inputs (angles)
        # TorchLayer will internally handle creating/using trainable weights.
        # Ensure angles is float32 (pennylane + torch prefers float32)
        angles = angles.to(dtype=torch.get_default_dtype())

        # Pass through quantum layer -> returns tensor [B, n_qubits]
        qout = self.qlayer(angles)

        # Map back to latent dimension
        zq = self.post_map(qout)

        return zq
