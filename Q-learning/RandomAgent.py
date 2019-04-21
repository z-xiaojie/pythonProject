import random
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class RandomAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        #
        self.switched = False
        # information for the game
        self.reward_history = [0]

    def add_reward(self, increment):
        self.reward_history.append(self.reward_history[-1] + increment)

    def act(self):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

    def reward(self, actions, selected_action, state):
        if state[selected_action] == 1:
            return 0
        else:
            selected = 0
            for i in range(len(actions)):
                if actions[i] == selected_action:
                    selected += 1
            return 1.0 / selected


