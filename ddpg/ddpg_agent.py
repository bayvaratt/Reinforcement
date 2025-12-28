import torch
from ddpg.actor_network import Actor
from ddpg.critic_network import Critic
from ddpg.noise import OUNoise
import numpy as np
from replays.replay_buffer import ReplayBuffer
import os

class DDPGAgent:
    """
    DDPG agent containing actor, critic, target networks, and update logic.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount_factor: float = 0.99, # discount factor
        soft_update_factor: float = 0.005, # target network update rate
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.discount_factor = discount_factor
        self.soft_update_factor = soft_update_factor
        self.max_action = max_action
        self.device = device
        self.criterion = torch.nn.MSELoss()

        """ Actor initialisation: actor, target actor, optimiser"""
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim=400).to(self.device)

        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim=400).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        """ Critic initialisation: critic, target critic, optimiser"""
        self.critic = Critic(state_dim, action_dim, hidden_dim=400).to(self.device)

        self.critic_target = Critic(state_dim, action_dim, hidden_dim=400).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        """ Exploration noise object"""
        self.noise = OUNoise(action_dim)
        self.noise_scale = 1.0  # Initial noise scaling factor
        self.noise_decay = 0.995  #Decay factor per episode
        self.min_noise_scale = 0.1  #Minimum noise level

    def select_action(self, state: np.ndarray, noise: bool = True) -> np.ndarray:
        """
        Select deterministic action from actor.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action_tensor = self.actor(state_tensor)
        self.actor.train()
        action = action_tensor.cpu().data.numpy().flatten()
        if noise:
            action += self.noise.sample() * self.max_action * self.noise_scale
        return np.clip(action, -self.max_action, self.max_action) # Clip action to bounds (unnecessary if env normalises to [-1,1])
    
    def decay_noise(self) -> None:
        """
        Decay the noise scale factor which is called at the end of each episode.
        """
        self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)
    
    def get_sample_batch(self, her_buffer: ReplayBuffer, batch_size: int):
        batch = her_buffer.sample(batch_size) # Get sample batch from HER buffer as a dict containing states, actions, rewards, next_states, dones as keys

        state = torch.FloatTensor(batch['states']).to(self.device)
        action = torch.FloatTensor(batch['actions']).to(self.device)
        reward = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(batch['next_states']).to(self.device)
        done = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device) # used to measure episode termination (1 if done, 0 otherwise)
        
        return state, action, reward, next_state, done

    def train(self, her_buffer: ReplayBuffer, batch_size: int) -> None:
        """
        Perform one DDPG training step
        (critic update -> actor update -> target update).
        """
        state, action, reward, next_state, done = self.get_sample_batch(her_buffer, batch_size)

        #Compute target Q-values
        with torch.no_grad():
            chosen_next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, chosen_next_action)
            target_Q = reward + (1 - done) * self.discount_factor * target_Q # derived from Bellman equation: Q(s,a) = r + gamma * Q'(s', a')

        # Update critic network using backpropagation to minimise MSE loss between current Q and target Q-values
        current_Q = self.critic(state, action)
        critic_loss = self.criterion(current_Q, target_Q)
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()


        # Update actor network using backpropagation: We negate Q-values to perform gradient ascent to maximise reward
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        # Update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, source_net: torch.nn.Module, target_net: torch.nn.Module) -> None:
        """
        Soft-update target network parameters dervied from eq: theta_target = tau*theta_source + (1 - tau)*theta_target).
        """
        for p, target_p in zip(source_net.parameters(), target_net.parameters()):
            target_p.data.copy_(self.soft_update_factor * p.data + (1 - self.soft_update_factor) * target_p.data)

            

    def save(self, filepath: str) -> None:
        """
        Save model and optimiser parameters.
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimiser_state_dict': self.actor_optimiser.state_dict(),
            'critic_optimiser_state_dict': self.critic_optimiser.state_dict(),
        }, filepath)

    def load(self, filepath: str) -> None:
        """
        Load model and optimiser parameters.
        """
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.actor_optimiser.load_state_dict(checkpoint['actor_optimiser_state_dict'])
            self.critic_optimiser.load_state_dict(checkpoint['critic_optimiser_state_dict'])
