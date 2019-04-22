import numpy as np
import random
import torch
from DDQN import DDQNetwork
from DQN import DQNetwork
from DQNAgent import DQNAgent
from RandomAgent import RandomAgent

INIT_PARAM = {
    "SEED": 15,
    "NUM_LAYERS": 2,
    "BATCH_SIZE": 32,
    "FC1_UNITS": 100,
    "FC2_UNITS": 100,
    "A_UNITS": 10,
    "V_UNITS": 10
}


def agent_create(state_size, action_size, a_type):
    if a_type == "RANDOM":
        return RandomAgent(state_size, action_size)
    else:
        if a_type == "DQN":
            local_dqn = DQNetwork(state_size, action_size, INIT_PARAM)
            target_dqn = DQNetwork(state_size, action_size, INIT_PARAM)
        elif a_type == "DDQN":
            local_dqn = DDQNetwork(state_size, action_size, INIT_PARAM)
            target_dqn = DDQNetwork(state_size, action_size, INIT_PARAM)
        else:
            return "ERROR"
        return DQNAgent(state_size, action_size, INIT_PARAM["SEED"], local_dqn, target_dqn)
    return "ERROR"
