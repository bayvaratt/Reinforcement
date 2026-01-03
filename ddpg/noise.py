import numpy as np

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(len(self.state))
        self.state = self.state + dx
        return self.state