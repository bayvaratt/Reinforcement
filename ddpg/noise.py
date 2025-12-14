import numpy as np


class OUNoise:
    """
    Ornstein–Uhlenbeck process for exploration noise.
    """

    def __init__(self, action_dim: int, mu=0.0, theta=0.15, sigma=0.2):
        pass

    def reset(self):
        """
        Reset internal noise state (called at start of each episode).
        """
        pass

    def sample(self) -> np.ndarray:
        """
        Sample noise vector.

        Returns:
            noise: np.ndarray of shape (action_dim,)
        """
        pass
