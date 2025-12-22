import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Maps state (including goal) -> continuous action.
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim # Input dimension (state + goal)
        self.action_dim = action_dim # Output dimension (action)
        self.max_action = max_action # Maximum action value (for scaling - used in forward pass)
        
        self.network = nn.Sequential( # state_dim -> 400 (hidden layer size) -> 400 -> action_dim
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
        action = self.network(state)
        return action * self.max_action
