import numpy as np


class ReplayBuffer:
    """
   Replay Buffer for storing and sampling transitions - HER implemented in training loop (trainer.py).
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Buffer structure: Each key holds an array with a fixed capacity to store respective elements of transitions
        self.buffer = {
            "states": np.zeros((capacity, state_dim), dtype=np.float32),
            "actions": np.zeros((capacity, action_dim), dtype=np.float32),
            "rewards": np.zeros((capacity,), dtype=np.float32),
            "next_states": np.zeros((capacity, state_dim), dtype=np.float32),
            "dones": np.zeros((capacity,), dtype=np.float32),
        }

        self.size = 0
        self.pointer = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition in the buffer.
        """
        index = self.pointer % self.capacity # Circular buffer: for example, if capacity=1000, after 1000 additions, start overwriting from index 0
        self.buffer["states"][index] = state
        self.buffer["actions"][index] = action
        self.buffer["rewards"][index] = reward
        self.buffer["next_states"][index] = next_state
        self.buffer["dones"][index] = done

        self.pointer += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """
        Sample a batch of transitions
        """
        indices = np.random.choice(self.size, size=batch_size, replace=False)
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