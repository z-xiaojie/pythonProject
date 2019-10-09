class Node:
    def __init__(self, cur_time=0):
        self.cur_time = cur_time

    def step(self, cur_time):
        self.cur_time = cur_time
