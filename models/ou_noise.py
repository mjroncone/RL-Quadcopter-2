import numpy as np
import copy

# Ornstein-Uhlenbeck noise generator based on: https://classroom.udacity.com/nanodegrees/nd009t/parts/ac12e0fe-e54e-40d5-b0f8-136dbdd1987b/modules/691b7845-f7d8-413d-90c7-971cd5016b5c/lessons/fef7e79a-0941-460b-936c-d24c759ff700/concepts/3178714d-0121-46e5-964e-bdcad3cdbe06

class OU_Noise():
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
