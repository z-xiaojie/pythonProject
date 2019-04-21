import keras
import numpy as np
import pandas as pd
import scipy.io as sio
import random
import torch
from torch import cuda
import matplotlib.pyplot as plt
from scipy.stats import genpareto

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Environment:
    def __init__(self, agents, dimension=12):
        #self.create_channel()
        #ch_state = pd.read_csv("/home/xiaojie/PycharmProjects/spectrum/Q-learning-network/train_state.csv",header=None)
        self.data = self.create_data(dimension=dimension)
        #self.data = np.array(self.data, dtype=float)
        #self.data = torch.from_numpy(self.data).to(device)
        #self.data = []
        #for i in range(len(ch_state.columns)):
            #self.data.append(ch_state[ch_state.columns[i]])
        self.dimension = len(self.data)
        self.state_size = self.dimension * 2 + 1
        self.action_size = self.dimension
        self.t = 0
        self.t_max = 0
        for i in range(self.dimension):
            if self.t_max < len(self.data[i]):
                self.t_max = len(self.data[i])
        self.number_of_agent = agents

    def temp_summary(self, agents):
        total = 0
        for t in range(self.t):
            valid = 0
            for col in range(self.dimension):
                if self.data[col][t] == 0:
                    valid += 1
            if valid > agents:
                valid = agents
            total += valid / agents
        return round(total/self.t, 3)

    def summary(self):
        for col in range(self.dimension):
            valid = self.t_max - np.sum(self.data[col][:self.t_max])
            print("resource=",col, "total valid=", valid, str(round(100*valid/self.t_max,4))+"%")
        max = 0
        for t in range(self.t_max):
            valid = 0
            for col in range(self.dimension):
                if self.data[col][t] == 0:
                    valid = 1
                    break
            max = max + valid
        total = 0
        for col in range(self.dimension):
            total += np.sum(self.data[col][:1000])
        print("max", max)

    def generate_next_state(self, env_state, action, reward):
        action_vector = keras.utils.to_categorical(action, self.action_size)
        reward_vector = [reward]
        next_state = np.concatenate((env_state, action_vector, reward_vector), axis=None)
        return np.reshape(next_state, [1, self.state_size]).astype(float)

    def process_utility_state(self, actions, env_state):
        utility_state = []
        for i in range(len(env_state)):
           if env_state[i] == 1:
               utility_state.append(0)
           else:
               selected_i = 0
               for j in range(len(actions)):
                   if actions[j] == i:
                       selected_i += 1
               if selected_i > 0:
                   utility_state.append(round(1 / selected_i,3))
               else:
                   utility_state.append(1)
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

    def process_state(self, action, env_state, reward):
        # action = [0 0 0 0 0 0 1 0 0 0 0 0]
        # env = [0 0 0 0 0 0 1 0 0 0 0 0]
        # migrating = [0]
        # reward = [0]
        # state = [migrating, action, env, reward]
        action_vector = keras.utils.to_categorical(action, self.action_size)
        state = np.concatenate((action_vector, env_state, [reward]), axis=None)
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

    def statistic(self, data_source, mean, frequency_selected, CH_state):
        mat_contents = sio.loadmat(data_source + ".mat")
        measurement_results = np.array(mat_contents["measurementResults"])
        for sweep in range(measurement_results.shape[0]):
            for i in range(len(frequency_selected)):
                if measurement_results[sweep][frequency_selected[i]] > mean:
                    CH_state[i].append(1)
                else:
                    CH_state[i].append(0)

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

    def create_channel(self, type='create', threshold=-107):
        if type == 'create':
            # how to select the channels
            number_ch = np.random.randint(15) + 5
            # start_f = np.random.randint(low=300, high=8000)
            frequency_selected = random.sample(range(0, 8192), number_ch)
            CH_power = []
            CH_state = []
            size = 150
            for i in range(number_ch):
                CH_power.append([])
                CH_state.append([])
            print("create ", number_ch, "channels, with frequency ", frequency_selected)
            for i in range(1, size, 1):
                if i < 10:
                    self.statistic("data/02_NE/0770MHz/MeasRes_0770_000" + str(i), threshold, frequency_selected, CH_state)
                elif 10 <= i < 100:
                    self.statistic("data/02_NE/0770MHz/MeasRes_0770_00" + str(i), threshold, frequency_selected, CH_state)
                else:
                    self.statistic("data/02_NE/0770MHz/MeasRes_0770_0" + str(i), threshold, frequency_selected, CH_state)
                print("done " + str(i))
            CH_power = np.asarray(CH_power).astype(int)
            CH_state = np.asarray(CH_state).astype(int)
            np.savetxt("train_power.csv", np.transpose(CH_power), delimiter=",", fmt='%i')
            np.savetxt("train_state.csv", np.transpose(CH_state), delimiter=",", fmt='%i')
        return  number_ch
