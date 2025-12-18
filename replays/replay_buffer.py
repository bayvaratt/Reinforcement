import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        pass

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store a transition in the buffer.
        """
        pass

    def sample(self, batch_size: int) -> dict:
        """
        Sample a batch of transitions.

        Returns:
            batch: dict containing states, actions, rewards, next_states, dones
        """
        pass

    def __len__(self) -> int:
        """
        Current number of transitions stored.
        """
        pass
