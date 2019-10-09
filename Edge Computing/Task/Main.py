import numpy as np
from Task.EdgeCloud import EdgeCloud

"""
network_param = {
   "avg_data": [],
   "interval": 33
}
"""


class Main(EdgeCloud):
    def __init__(self, m_id, network_param, assigned_users, number_of_core=2, bandwidth_max=1):
        super(Main, self).__init__(0, assigned_users, number_of_core, bandwidth_max)
        self.avg_data = network_param["avg_data"]
        self.interval = network_param["interval"]
        self.ch = network_param["ch"]
        # id
        self.id = m_id

        #
        #self.avg_data[0] = 0.8

    def summary(self):
        core_density = np.zeros(self.number_of_core)
        core_utility = np.zeros(self.number_of_core)
        for item in self.user_to_core:
            user = self.find_user_by_id(item["user_id"])
            if user is not None:
                core_density[item["core_id"]] += user.get_density_with_rate()
                core_utility[item["core_id"]] += user.get_utility_with_rate()
        return core_density, core_utility

    def display(self):
        core_density, core_utility = self.summary()
        """
        print("edge cloud", self.id, self.user_to_core)
        print("allocated data rate", self.users[0].rate)
        for n in range(self.get_number_of_users()):
            self.users[n].summary()
        """
        print("utility", round(np.sum(core_utility), 4), "core utility", core_utility,
              "density", round(np.sum(core_density), 4), "core density", core_density,
              "complete:", self.complete_history[-1])


def create_edge(evn, number_of_edge, avg_data=None, time_interval=33, channel=None, number_of_core=2, bandwidth_max=None):
    edge = list()
    for k in range(number_of_edge):
        edge.append(Main(k, {
            "avg_data": avg_data[k],
            "interval": time_interval,
            "ch": channel[k]
        }, None, number_of_core=number_of_core, bandwidth_max=bandwidth_max))
    return edge

