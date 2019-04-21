import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
FC1_UNITS = 100
FC2_UNITS = 100
FC3_UNITS = 75
FC4_UNITS = 75
A_UNITS = 75
V_UNITS = 75


class DDQNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, num_layers=1, batch_size=BATCH_SIZE):
        # Initialize parameters and build model.
        super(DDQNetwork, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        # layers
        self.fc1 = nn.Linear(self.state_size, FC1_UNITS)
        self.fc2 = nn.Linear(FC1_UNITS, FC2_UNITS)
        self.fc3 = nn.Linear(FC2_UNITS, FC3_UNITS)
        self.fc4 = nn.Linear(FC3_UNITS, FC4_UNITS)
        # action in given state advantage + value of the state
        self.a_1 = nn.Linear(FC4_UNITS, A_UNITS)
        self.v_1 = nn.Linear(FC4_UNITS, V_UNITS)
        self.a_2 = nn.Linear(A_UNITS, self.action_size)
        self.v_2 = nn.Linear(V_UNITS, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # add Q(s,a) = v + a
        x = x.view(x.size(0), -1)
        a = F.relu(self.a_1(x))
        v = F.relu(self.v_1(x))
        a = self.a_2(a)
        v = self.v_2(v).expand(x.size(0), self.action_size)
        x = v + a - a.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x


class DQNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, num_layers=1, batch_size=BATCH_SIZE):

        super(DQNetwork, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        # layers
        self.fc1 = nn.Linear(self.state_size, FC1_UNITS)
        self.fc2 = nn.Linear(FC1_UNITS, FC2_UNITS)
        self.fc3 = nn.Linear(FC2_UNITS, FC3_UNITS)
        self.fc4 = nn.Linear(FC3_UNITS, FC4_UNITS)
        self.fc5 = nn.Linear(FC4_UNITS, self.action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class LDDQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, num_layers=2, batch_size=BATCH_SIZE):
        # Initialize parameters and build model.
        super(LDDQNetwork, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        # layers
        self.fc1 = nn.Linear(self.state_size, FC1_UNITS)
        self.fc2 = nn.Linear(FC1_UNITS, FC2_UNITS)
        self.fc3 = nn.Linear(FC2_UNITS, FC3_UNITS)
        self.fc4 = nn.Linear(FC3_UNITS, FC4_UNITS)
        self.lstm = nn.LSTM(input_size=FC4_UNITS, hidden_size=FC4_UNITS, num_layers=self.num_layers)
        # action in given state advantage + value of the state
        self.a_1 = nn.Linear(FC4_UNITS, A_UNITS)
        self.v_1 = nn.Linear(FC4_UNITS, V_UNITS)
        self.a_2 = nn.Linear(A_UNITS, self.action_size)
        self.v_2 = nn.Linear(V_UNITS, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        _out, _ = self.lstm(x.view(len(x), 1, -1))
        # add Q(s,a) = v + a
        x = x.view(_out.size(0), -1)
        a = F.relu(self.a_1(x))
        v = F.relu(self.v_1(x))
        a = self.a_2(a)
        v = self.v_2(v).expand(x.size(0), self.action_size)
        x = v + a - a.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x
