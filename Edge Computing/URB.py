from operator import attrgetter
from Check import condition_one, condition_two, total_utility_density
import pandas as pd
import numpy as np
from Node import Node


class URB(Node):
    def __init__(self, cur_time, data, interval=33, ch=[11]):
        super(URB, self).__init__(cur_time)

        # decision
        self.user_to_core = []

        # monitor environment
        self.avg = data
        self.interval = interval

