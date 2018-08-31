import numpy as np
from task import Task
from models.replay_buffer import ReplayBuffer
from models.actor import Actor
from models.critic import Critic
from models.ou_noise import OU_Noise

# Implementation of the DDPG algorithm described here: https://arxiv.org/pdf/1509.02971.pdf
# Based on implementation here: https://classroom.udacity.com/nanodegrees/nd009t/parts/ac12e0fe-e54e-40d5-b0f8-136dbdd1987b/modules/691b7845-f7d8-413d-90c7-971cd5016b5c/lessons/fef7e79a-0941-460b-936c-d24c759ff700/concepts/6fc1ab59-bed7-49f2-990d-16505353f9bc#

class DDPG_Agent():
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Syncing target model weights to local model weights to start
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Process for creating noise
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OU_Noise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # ReplayBuffer settings
        self.buffer_size = 100000
        self.batch_size = 64
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size)

        self.score = 0
        self.best_score = -np.inf
        self.gamma = 0.99
        self.tau = 0.01


    def reset_episode(self):
        self.total_reward = 0
        self.count = 0

        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        self.buffer.add(self.last_state, action, reward, next_state, done)
        self.count += 1
        self.total_reward += reward

        if len(self.buffer) > self.batch_size:
            experiences = self.buffer.sample()
            self.learn(experiences)

        self.last_state = next_state

    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())

    def learn(self, experiences):
        self.score = self.total_reward / float(self.count) if self.count else 0
        if self.score > self.best_score:
            self.best_score = self.score

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_function([states, action_gradients, 1])

        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), 'Local and target model parameters must be of the same size'

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
