import keras
import numpy as np
import pandas as pd
import scipy.io as sio
import random
import torch
from torch import cuda
import matplotlib.pyplot as plt
from scipy.stats import genpareto

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Environment:
    def __init__(self, agents, dimension=12, migration_overhead=0, data="old", service="same"):

        self.data = []
        if data == "old":
            ch_state = pd.read_csv("train_state.csv", header=None)
            self.dimension = len(ch_state.columns)
            for i in range(len(ch_state.columns)):
                self.data.append(ch_state[ch_state.columns[i]])
        elif data == "new":
            self.data = self.create_data(dimension=dimension)

        self.migration_overhead = migration_overhead
        self.dimension = len(self.data)
        self.state_size = self.dimension * 2 + 2 + migration_overhead + 1
        self.action_size = self.dimension

        self.t = 0
        self.t_max = len(self.data[0])
        for i in range(self.dimension):
            if self.t_max > len(self.data[i]):
                self.t_max = len(self.data[i])
        for i in range(self.dimension):
            self.data[i] = self.data[i][:self.t_max]
        self.number_of_agent = agents

        self.data = torch.from_numpy(np.array(self.data)).float().to(device)

        self.QoS = torch.zeros(self.dimension).float().to(device)
        if service == "same":
            for i in range(self.dimension):
                self.QoS[i] = 1
        else:
            for i in range(self.dimension):
                self.QoS[i] = round((np.random.randint(10) + 1) / 10, 1)

    def temp_summary(self, agents):
        total = 0
        for t in range(self.t):
            valid = 0
            for col in range(self.dimension):
                if self.data[col][t] == 0:
                    valid += self.QoS[col]
            if valid > agents:
                valid = agents
            total += valid
        return total.round()

    def summary(self):
        for col in range(self.dimension):
            valid = self.t_max - self.data[col][:self.t_max].sum()
            valid = 100*valid/self.t_max
            print("resource=", col, "valid=", valid, "%")
        print("duration", self.t_max)
        print("QoS", self.QoS)

    def generate_next_state(self, env_state, action, reward):
        action_vector = keras.utils.to_categorical(action, self.action_size)
        reward_vector = [reward]
        next_state = np.concatenate((env_state, action_vector, reward_vector), axis=None)
        return np.reshape(next_state, [1, self.state_size]).astype(float)

    def process_utility_state(self, agents, actions, env_state):
        utility_state = torch.zeros(len(env_state))
        for i in range(len(env_state)):
            if env_state[i] == 1:
                utility_state[i] = 0
            else:
                selected_i = 0
                for j in range(len(actions)):
                    if actions[j] == i and agents[j].migration_overhead == 0:
                        selected_i += 1
                if selected_i > 0:
                    utility_state[i] = (self.QoS[i] / selected_i)
                else:
                    utility_state[i] = self.QoS[i]
        return utility_state

    def add_time(self, increment):
        self.t += increment

    def step(self):
        self.t += 1

    def reset_time(self):
        self.t = 0

    def time(self):
        return self.t

    def end_of_game(self):
        self.t += 1
        if self.t > self.t_max:
            return True
        return False

    def get_env_state(self):
        state = []
        for i in range(self.dimension):
            state.append(self.data[i][self.t])
        return state

    def get_state_action_size(self):
        return self.state_size, self.action_size

    def process_state(self, migrating, overhead, action, env_state, reward):
        # action = [0 0 0 0 0 0 1 0 0 0 0 0]
        # env = [0 0 0 0 0 0 1 0 0 0 0 0]
        # migrating = [0]
        # reward = [0]
        # state = [migrating, action, env, reward]
        overhead_vector = keras.utils.to_categorical(overhead, self.migration_overhead + 1)
        action_vector = keras.utils.to_categorical(action, self.action_size)
        if migrating:
            state = np.concatenate(([1], overhead_vector, action_vector, env_state, [reward]), axis=None)
        else:
            state = np.concatenate(([0], overhead_vector, action_vector, env_state, [reward]), axis=None)
        state = np.reshape(state, [1, self.state_size]).astype(float)
        return state

    def reward(self, actions, selected_action, state):
        if state[selected_action] == 1:
            return 0
        else:
            selected = 0
            for i in range(len(actions)):
                if actions[i] == selected_action:
                    selected += 1
            return 1.0 / selected

    def create_data(self, dimension):
        size = 55000
        busy_duration = []
        idle_duration = []
        for i in range(dimension):
            busy_duration.append(np.random.exponential(np.random.randint(30)+6, size=size).astype(int))
            idle_duration.append(np.random.exponential(np.random.randint(30)+6, size=size).astype(int))
        CH = []
        for i in range(dimension):
            s = []
            for d in range(size):
                for b in range(busy_duration[i][d]):
                    s.append(1)
                for b in range(idle_duration[i][d]):
                    s.append(0)
            CH.append(s)
        return CH
