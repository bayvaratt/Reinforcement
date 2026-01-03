import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Maps state (including goal) to continuous action.
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: torch.Tensor, hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        self.max_action = max_action 
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the actor network.
        """
        x = self.network(state)  # Pass through layers
        return x * self.max_action
