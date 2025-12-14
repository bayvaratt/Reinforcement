import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Deterministic policy network.
    Maps state (including goal) -> continuous action.
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        pass

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the actor network.

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            action: Tensor of shape (batch_size, action_dim)
        """
        pass
