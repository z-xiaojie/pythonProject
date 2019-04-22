import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, param):

        super(DQNetwork, self).__init__()
        self.num_layers = param["NUM_LAYERS"]
        self.batch_size = param["BATCH_SIZE"]
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(param["SEED"])
        # layers
        self.fc1 = nn.Linear(self.state_size, param["FC1_UNITS"])
        self.fc2 = nn.Linear(param["FC1_UNITS"], param["FC2_UNITS"])
        #self.fc3 = nn.Linear(FC2_UNITS, FC3_UNITS)
        #self.fc4 = nn.Linear(FC3_UNITS, FC4_UNITS)
        self.fc5 = nn.Linear(param["FC2_UNITS"], self.action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        return self.fc5(x)

