import torch
import os
import sys
from typing import Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddpg.actor_network import Actor
from ddpg.critic_network import Critic
from ddpg.noise import OUNoise
import numpy as np
from replays.replay_buffer import ReplayBuffer


class DDPGAgent:
    """
    DDPG agent containing actor, critic, target networks, and update logic.
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: Union[float, np.ndarray], discount_factor: float = 0.99, soft_update_factor: float = 0.001, device: torch.device = torch.device("cpu"),):
        self.discount_factor = discount_factor
        self.soft_update_factor = soft_update_factor
        if isinstance(max_action, np.ndarray):
            self.max_action = torch.FloatTensor(max_action).to(device)
        else:
            self.max_action = torch.FloatTensor([max_action] * action_dim).to(device)
        self.device = device
        self.criterion = torch.nn.MSELoss()

        # Actor: network, target network, and optimiser
        self.actor = Actor(state_dim, action_dim, self.max_action, hidden_dim=400).to(self.device)

        self.actor_target = Actor(state_dim, action_dim, self.max_action, hidden_dim=400).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=5e-5)  # Fine-tune lr if needed

        # Critic: network, target network, and optimiser
        self.critic = Critic(state_dim, action_dim, hidden_dim=400).to(self.device)

        self.critic_target = Critic(state_dim, action_dim, hidden_dim=400).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=5e-4)

        # Exploration noise
        self.noise = OUNoise(action_dim)
        self.noise_scale = 0.5
        self.noise_decay = 0.995
        self.min_noise_scale = 0.05

    def select_action(self, state: np.ndarray, noise: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action_tensor = self.actor(state_tensor)
        self.actor.train()
        action = action_tensor.cpu().data.numpy().flatten()
        
        max_action_np = self.actor.max_action.cpu().numpy()
        
        if noise:
            action += self.noise.sample() * max_action_np * self.noise_scale  # Add exploration noise
        
        return np.clip(action, -max_action_np, max_action_np)

    def decay_noise(self) -> None:
        """
        Decay the noise scale factor which is called at the end of each episode.
        """
        self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)  # Exponential decay

    def get_sample_batch(self, her_buffer: ReplayBuffer, batch_size: int):
        batch = her_buffer.sample(batch_size)

        state = torch.FloatTensor(batch["states"]).to(self.device)
        action = torch.FloatTensor(batch["actions"]).to(self.device)
        reward = torch.FloatTensor(batch["rewards"]).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(batch["next_states"]).to(self.device)
        done = torch.FloatTensor(batch["dones"]).unsqueeze(1).to(self.device)

        return state, action, reward, next_state, done

    def train(self, her_buffer: ReplayBuffer, batch_size: int) -> None:
        """
        Perform one DDPG training step with gradient clipping.
        """
        state, action, reward, next_state, done = self.get_sample_batch(her_buffer, batch_size)

        # Compute target Q-values
        with torch.no_grad():
            chosen_next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, chosen_next_action)
            target_Q = (reward + (1 - done) * self.discount_factor * target_Q)

        # Update critic - minimise MSE loss between current Q and target Q
        current_Q = self.critic(state, action)
        critic_loss = self.criterion(current_Q, target_Q)
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Prevent exploding gradients
        self.critic_optimiser.step()

        # Update actor - negate the critic output to perform gradient ascent to maximise expected return
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimiser.step()

        # Update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, source_net: torch.nn.Module, target_net: torch.nn.Module) -> None:
        """
        Soft-update target network parameters derived from:
        theta_target = tau * theta_source + (1 - tau) * theta_target
        """
        for p, target_p in zip(source_net.parameters(), target_net.parameters()):
            target_p.data.copy_(self.soft_update_factor * p.data + (1 - self.soft_update_factor) * target_p.data)

    def save(self, filepath: str) -> None:
        """
        Save model and optimiser parameters.
        """
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimiser_state_dict": self.actor_optimiser.state_dict(),
                "critic_optimiser_state_dict": self.critic_optimiser.state_dict(),
            },
            filepath,
        )  # Save checkpoint to disk



    def load(self, filepath: str) -> None:
        """
        Load model and optimiser parameters.
        """
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.actor_optimiser.load_state_dict(checkpoint["actor_optimiser_state_dict"])
            self.critic_optimiser.load_state_dict(checkpoint["critic_optimiser_state_dict"])  # Restore optimiser states
