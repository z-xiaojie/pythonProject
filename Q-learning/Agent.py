import numpy as np
import random
import torch
from DDQN import DDQNetwork
from DQN import DQNetwork
from DQNAgent import DQNAgent
from RandomAgent import RandomAgent

INIT_PARAM = {
    "SEED": 15,
    "NUM_LAYERS": 1,
    "BATCH_SIZE": 32,
    "FC1_UNITS": 200,
    "FC2_UNITS": 200,
    "A_UNITS": 50,
    "V_UNITS": 50
}


def agent_create(state_size, action_size, a_type):
    if a_type == "RANDOM":
        return RandomAgent(a_type, state_size, action_size)
    else:
        if a_type == "DQN":
            local_dqn = DQNetwork(state_size, action_size, INIT_PARAM)
            target_dqn = DQNetwork(state_size, action_size, INIT_PARAM)
        elif a_type == "DDQN":
            local_dqn = DDQNetwork(state_size, action_size, INIT_PARAM)
            target_dqn = DDQNetwork(state_size, action_size, INIT_PARAM)
        else:
            return "ERROR"
        return DQNAgent(a_type, state_size, action_size, INIT_PARAM["SEED"], local_dqn, target_dqn)
    return "ERROR"
