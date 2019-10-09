from DAG import DAG
import random
import math
import json
import numpy as np


class EdgeCloud:
    def __init__(self, id, frequency, compute_id, transmission_power=0.5, bandwidth=None):
        self.freq = int(frequency)
        self.compute_id = compute_id
        self.DAG = None
        self.transmission_power = transmission_power
        self.bandwidth = bandwidth

        self.tasks = []
        self.channels = []
        self.adjust = 1

        self.id = id

        self.current_cjob = None

        # job queue
        self.queue = []

        #
        self.preference = None

        #
        self.density = 0

        #
        self.weight = 0

    def accept(self, task):
        self.tasks.append(task)

    def check(self):
        density, fail = self.summary()
        self.density = density
        return density <= 1 and not fail

    def summary(self):
        self.channel_allocation()
        density = 0
        fail = False
        for n in range(len(self.tasks)):
            rate = self.rate(self.tasks[n].transmission_power, self.tasks[n].H[self.id])
            self.tasks[n].remote_deadline = self.tasks[n].DAG.D / 1000 - self.tasks[n].local / self.tasks[n].freq - \
                                            self.tasks[n].local_to_remote_size / rate
            if self.tasks[n].remote_deadline < 0:
                fail = True
            density += (self.tasks[n].remote/self.freq) / np.min([self.tasks[n].remote_deadline, self.tasks[n].DAG.T / 1000])
        return density, fail

    def edge_density(self):
        self.channel_allocation()
        density = 0
        for n in range(len(self.tasks)):
            rate = self.rate(self.tasks[n].transmission_power, self.tasks[n].H[self.id])
            self.tasks[n].remote_deadline = self.tasks[n].DAG.D / 1000 - self.tasks[n].local / self.tasks[n].freq - \
                                            self.tasks[n].local_to_remote_size / rate
            if self.tasks[n].remote_deadline < 0:
                density += 1
            else:
                density += (self.tasks[n].remote / self.freq) / np.min(
                [self.tasks[n].remote_deadline, self.tasks[n].DAG.T / 1000])
        return density

    def get_density(self, task_id):
        for n in range(len(self.tasks)):
            if self.tasks[n].task_id == task_id:
                rate = self.rate(self.tasks[n].transmission_power, self.tasks[n].H[self.id])
                self.tasks[n].remote_deadline = self.tasks[n].DAG.D / 1000 - self.tasks[n].local / self.tasks[n].freq - \
                                            self.tasks[n].local_to_remote_size / rate
                density = (self.tasks[n].remote/self.freq) / np.min([self.tasks[n].remote_deadline, self.tasks[n].DAG.T / 1000])
                return round(density, 4)
        return 0

    def channel_allocation(self):
        number_of_up_links = len(self.tasks)
        self.channels = []
        for n in range(number_of_up_links):
            self.channels.append(self.bandwidth / number_of_up_links)

    def rate(self, transmission_power, H=math.pow(10, -3), N_0=math.pow(10, -9)):
        #print("len....", len(self.tasks), len(self.queue))
        r = self.channels[0] * math.pow(10, 6) * math.log2(1 + transmission_power * math.pow(H, 2) / N_0)
        #print(r, "bit/s", r/8000, "KB/s")
        return r

    def max_rate(self, transmission_power, H=math.pow(10, -3), N_0=math.pow(10, -9)):
        r = self.bandwidth * math.pow(10, 6) * math.log2(1 + transmission_power * math.pow(H, 2) / N_0)
        #print(r, "bit/s", r/8000, "KB/s")
        return r

    def update_bandwidth(self, current=False):
        if not current:
            return self.bandwidth / max(len(self.tasks)+1, 1)
        else:
            return self.bandwidth / max(len(self.tasks), 1)

    def priority(self, total_bandwidth, total_freq):
        a = 0.5
        b = 1 - a
        weight = a*self.update_bandwidth()/total_bandwidth + b*self.freq/total_freq
        if self.weight == 0:
            self.weight = weight
        if not self.check():
            return 0
        return weight

    def get_top_cjob(self):
        #if self.current_cjob is not None and self.queue.__contains__(self.current_cjob):
            #return self.current_cjob
        self.queue.sort(key=lambda x: x.deadtime)
        first = None
        for item in self.queue:
            if first is None and item.ready_exe():
                first = item
            elif item.ready_exe():
                item.queue_time += 1
        return first
        #self.current_cjob = None
        #return None

    def start_to_transmit(self, task_id, release_id, local_finished_time):
        for item in self.queue:
            if item.task_id == task_id and item.release_id == release_id:
                item.local_finished = True
                item.local_finished_time = local_finished_time

    def get_bandwidth(self, n):
        number_of_users = len(self.tasks)
        for item in self.tasks:
            if item.task_id == n:
                return self.bandwidth / number_of_users
        return self.bandwidth / (number_of_users + 1)






