import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DDQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, param):
        # Initialize parameters and build model.
        super(DDQNetwork, self).__init__()
        self.num_layers = param["NUM_LAYERS"]
        self.batch_size = param["BATCH_SIZE"]
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(param["SEED"])

        """
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
         def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()        
        
        t = 12000 0.745 0.509 
        
        """
        # layers
        self.fc1 = nn.Linear(self.state_size, param["FC1_UNITS"])
        self.fc2 = nn.Linear(param["FC1_UNITS"], param["FC2_UNITS"])
        #self.fc3 = nn.Linear(FC2_UNITS, FC3_UNITS)
        #self.fc4 = nn.Linear(FC3_UNITS, FC4_UNITS)
        self.lstm = nn.LSTM(input_size=param["FC2_UNITS"], hidden_size=param["FC2_UNITS"], num_layers=self.num_layers)
        # action in given state advantage + value of the state
        self.a_1 = nn.Linear(param["FC2_UNITS"], param["A_UNITS"])
        self.v_1 = nn.Linear(param["FC2_UNITS"], param["V_UNITS"])
        self.a_2 = nn.Linear(param["A_UNITS"], self.action_size)
        self.v_2 = nn.Linear(param["V_UNITS"], 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        # add Q(s,a) = v + a
        _out, _ = self.lstm(x.view(len(x), 1, -1))
        x = x.view(x.size(0), -1)
        a = F.relu(self.a_1(x))
        v = F.relu(self.v_1(x))
        a = self.a_2(a)
        v = self.v_2(v).expand(x.size(0), self.action_size)
        x = v + a - a.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x
