import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    Q-function approximator.
    Maps (state, action) -> scalar Q-value.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        pass

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.

        Args:
            state: Tensor of shape (batch_size, state_dim)
            action: Tensor of shape (batch_size, action_dim)

        Returns:
            q_value: Tensor of shape (batch_size, 1)
        """
        pass
