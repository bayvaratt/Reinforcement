import torch
from ddpg.actor_network import Actor
from ddpg.critic_network import Critic


class DDPGAgent:
    """
    DDPG agent containing actor, critic, target networks,
    and update logic.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        pass

    def select_action(self, state):
        """
        Select deterministic action from actor.

        Args:
            state: np.ndarray of shape (state_dim,)

        Returns:
            action: np.ndarray of shape (action_dim,)
        """
        pass

    def train(self, replay_buffer, batch_size: int):
        """
        Perform one DDPG training step
        (critic update, actor update, target update).
        """
        pass

    def _soft_update(self, source_net, target_net):
        """
        Soft-update target network parameters.
        """
        pass
