import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Q-function approximator.
    Maps (state, action) -> scalar Q-value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)  # Concat state + action
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.
        """

        x = F.relu(self.ln1(self.fc1(state)))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x)
        return q_value
