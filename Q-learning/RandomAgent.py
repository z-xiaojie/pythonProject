import random
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class RandomAgent:
    def __init__(self, a_type, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.a_type = a_type
        # cost for action
        self.migration_overhead = 0
        self.switched = False
        # information for the game
        self.reward_history = [0]

    def add_reward(self, increment):
        self.reward_history.append(self.reward_history[-1] + increment)



