import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer with efficient pre-allocation.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store a transition in the buffer. Overwrites old data if full.
        """
        index = self.ptr % self.capacity

        self.states[index] = state
        self.actions[index] = action
        self.next_states[index] = next_state
        self.rewards[index] = reward
        self.dones[index] = done

        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """
        Sample a batch of transitions uniformly.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size