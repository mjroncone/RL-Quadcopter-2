import random
from collections import namedtuple, deque

# Based on the replay buffer outlined here: https://classroom.udacity.com/nanodegrees/nd009t/parts/ac12e0fe-e54e-40d5-b0f8-136dbdd1987b/modules/691b7845-f7d8-413d-90c7-971cd5016b5c/lessons/fef7e79a-0941-460b-936c-d24c759ff700/concepts/cfa43ab2-37dd-460a-ac08-3fb82f574749

class ReplayBuffer:
    """Stores experience tuples in a queue with fixed size"""

    def __init__(self, max_length, default_batch_size = 64):
        self.buffer = deque(maxlen = max_length)
        self.default_batch_size = default_batch_size
        self.experience_tuple = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        experience = self.experience_tuple(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size = None):
        batch_size = batch_size or self.default_batch_size
        return random.sample(self.buffer, k = batch_size)
