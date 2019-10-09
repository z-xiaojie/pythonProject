from Task.Job import Job
from Task.Node import Node
import numpy as np


class User(Node):
    'User'
    def __init__(self, cur_time, m_id, exe_time, job_size, interval, deadline):
        super(User, self).__init__(cur_time)
        self.id = m_id
        self.exe_time = exe_time
        self.job_size = job_size
        self.interval = interval
        self.deadline = deadline
        self.serial_number = 0

        # release control
        self.release_timer = interval
        self.jobs = list()

        # transmission information
        self.rate = 0
        """
        self.avg_tf = (self.job_size - 1)
        self.std_tf = 0
        self.actual_deadline = self.deadline - self.avg_tf
        self.actual_interval = self.interval + self.std_tf
        """
        #self.avg_tf = 0
        #self.cost = 0
        #self.density = self.get_density()
        #self.utility = self.get_utility()

        self.policy = None

    """
    def update_avg_tf(self, rate):
        self.avg_tf = round((self.job_size - 1)/ rate,3)
        self.get_density()
        self.get_utility()
    """
    def set_allocated_rate(self, rate):
        self.rate = rate

    # policy = 5 , only return original value
    def get_density_with_rate(self):
        if self.policy == 5:
            return self.exe_time/np.min([self.interval, self.deadline])
        else:
            if self.get_deadline_with_rate() > 0:
                density = self.exe_time / np.min(
                    [max(self.interval, self.job_size / self.rate), self.get_deadline_with_rate()])
            else:
                density = 1.5
            return round(density, 4)

    def get_utility_with_rate(self):
        if self.policy == 5:
            return self.exe_time/self.interval
        else:
            utility = self.exe_time / max(self.interval, self.job_size / self.rate)
            return round(utility, 4)

    # k is cloud, m is core
    def release_job(self, k,  m):
        if self.release_timer > 0:
            self.release_timer -= 1
            return 0
        if self.release_timer == 0:
            self.release_timer = self.interval
            self.serial_number += 1
            job = Job(self.cur_time, self.exe_time, self.job_size, self.interval, self.deadline,
                      self.cur_time + self.deadline, self.id, self.serial_number, k, m)
            self.add_job_to_queue(job)
            return 1

    def add_job_to_queue(self, job):
        self.jobs.append(job)

    def transmit_job(self):
        if len(self.jobs) > 0:
            job = self.jobs[0]
            while self.cur_time > job.dead_time:
                self.jobs.remove(job)
                if len(self.jobs) == 0:
                    return "None", None, 0
                else:
                    job = self.jobs[0]
            state, overflow = job.transmit(self.rate, self.cur_time)
            if state == "Transmitted":
                self.jobs.remove(job)
                return "Transmitted", job, overflow
        return "None", None, 0

    def simple_summary(self):
        print("[", self.exe_time, ",", self.job_size, ",", self.interval, ",", self.deadline,"]")

    def summary(self):
        print("exe_time:", self.exe_time, "job_size:", self.job_size, "interval:", self.interval, "deadline:",
              self.deadline, "density:", self.get_density_with_rate(), "utility:", self.get_utility_with_rate())

    def get_deadline_with_rate(self):
        if self.policy == 5:
            return self.deadline
        else:
            if self.rate == 0:
                data_tf = 9999
            else:
                data_tf = self.job_size / self.rate
            # when data transmit > interval
            waiting_in_local_queue = max(0, data_tf - self.interval)
            return max(0, round(self.deadline - data_tf - waiting_in_local_queue, 4))

