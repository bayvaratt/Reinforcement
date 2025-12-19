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

        #(1)
        self.fc1 = nn.Linear(state_dim, hidden_dim) # First layer processes state input
        self.ln1 = nn.LayerNorm(hidden_dim)

        #(2)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim) # Second layer processes state features, action concatenated
        self.ln2 = nn.LayerNorm(hidden_dim)

        #(3)
        self.fc3 = nn.Linear(hidden_dim, 1) # Output layer produces scalar Q-value

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.

        Args:
            state: Tensor of shape (batch_size, state_dim)
            action: Tensor of shape (batch_size, action_dim)

        Returns:
            q_value: Tensor of shape (batch_size, 1)
        """

        x = F.relu(self.ln1(self.fc1(state))) #(1)
        x = torch.cat([x, action], dim=1) #(2)
        x = F.relu(self.ln2(self.fc2(x))) #(2)
        q_value = self.fc3(x) #(3)
        return q_value
