from collections import deque
import random

import numpy as np


class QMemoryItem:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __repr__(self):
        return f"QMemoryItem(state={self.state}, action={self.action}, reward={self.reward}, next_state={self.next_state}, done={self.done})"

class Memory:
    def __init__(self, capacity=10000):
        # paměť je dvoukoncovou fronta
        # maxlen je maximální velikost paměti, pokud je paměť plná, nejstarší prvek se odstraní
        self.buffer = deque(maxlen=capacity)

    def push(self, item):
        if not isinstance(item, QMemoryItem):
            raise ValueError("Vstup do memory.push() musí být instance QMemoryItem")
        self.buffer.append(item)

    def sample(self, sample_size):
        # vyber náhodný vzorek z paměti o velikosti sample_size
        batch = random.sample(self.buffer, sample_size)
        states, actions, rewards, next_states, dones = zip(*[
            (item.state, item.action, item.reward, item.next_state, item.done) for item in batch
        ])
        # vrat jako np.array
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)