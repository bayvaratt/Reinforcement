import numpy as np


class OUNoise:
    """
    Ornstein–Uhlenbeck process for exploration noise.

    Used to generate temporally correlated noise for continuous action spaces.

    Parameters
    ----------
    action_dim : int
        Dimensionality of the action space.
    mu : float, optional
        Long-running mean (recommended 0.0).
    theta : float, optional
        Speed of mean reversion (recommended 0.15).
    sigma : float, optional
        Volatility parameter (recommended 0.2).
    dt : float, optional
        Time step for discretisation (recommended 1e-2).
    """

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2) -> None:
        """
        Initialise the OU noise process.

        Attributes Initialised:
            mu: target mean per action dimension.
            theta: mean reversion speed.
            sigma: volatility.
            action_dim: action dimensionality.
            state: current internal state (initialised via reset()).
        """
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        """
        Reset internal noise state (called at start of each episode) to long-running mean.
        """
        self.state = np.copy(self.mu)

    def sample(self) -> np.ndarray:
        """
        Corrected OU sampling step.
        """

        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        
        self.state = self.state + dx
        return self.state # May need to clip externally based on action bounds

