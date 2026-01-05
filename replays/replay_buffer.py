import numpy as np


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions.
    Hindsight Experience Replay (HER) is implemented in the training loop (trainer.py).
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Buffer structure: each key holds an array with fixed capacity
        self.buffer = {
            "states": np.zeros((capacity, state_dim), dtype=np.float32),
            "actions": np.zeros((capacity, action_dim), dtype=np.float32),
            "rewards": np.zeros((capacity,), dtype=np.float32),
            "next_states": np.zeros((capacity, state_dim), dtype=np.float32),
            "dones": np.zeros((capacity,), dtype=np.float32),
        }

        self.size = 0
        self.pointer = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the buffer.
        """
        index = self.pointer % self.capacity  # Circular buffer logic
        self.buffer["states"][index] = state
        self.buffer["actions"][index] = action
        self.buffer["rewards"][index] = reward
        self.buffer["next_states"][index] = next_state
        self.buffer["dones"][index] = done

        self.pointer += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """
        Sample a batch of transitions from the buffer.
        """
        # Ensure we don't sample more than available
        sample_size = min(batch_size, self.size)
        replace = sample_size > self.size
        
        indices = np.random.choice(self.size, size=sample_size, replace=replace)
        batch = {
            "states": self.buffer["states"][indices],
            "actions": self.buffer["actions"][indices],
            "rewards": self.buffer["rewards"][indices],
            "next_states": self.buffer["next_states"][indices],
            "dones": self.buffer["dones"][indices],
        }
        return batch
    
    def __len__(self) -> int:
        """
        Return the current size of the buffer.
        """
        return self.size