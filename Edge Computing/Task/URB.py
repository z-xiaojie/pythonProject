from Task.Node import Node


class URB(Node):
    def __init__(self, cur_time, data, interval=33, ch=[11]):
        super(URB, self).__init__(cur_time)

        # decision
        self.user_to_core = []

        # monitor environment
        self.avg = data
        self.interval = interval

