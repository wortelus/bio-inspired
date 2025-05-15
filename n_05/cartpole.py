import random

import gymnasium as gym


class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state = None
        self.done = False

    def get_dims(self):
        return {
            "state_dim": self.env.observation_space.shape[0],
            "action_dim": self.env.action_space.n
        }

    def reset(self):
        self.state, info = self.env.reset()
        self.done = False
        return self.state, info

    def step(self, action):
        self.state, reward, done, truncated, _ = self.env.step(action)
        self.done = done or truncated

        return self.state, reward, self.done

    def close(self):
        self.env.close()
