from DAG import DAG
import random
import math
import json
import numpy as np
import matplotlib.pyplot as plt


class Device:
    def __init__(self, frequency, task_id, H=None, transmission_power=0.5):
        self.freq = frequency
        self.task_id = task_id
        self.DAG = None
        self.p_max = transmission_power

        # local information
        self.k = math.pow(10, -28)

        self.H = H
        self.preference = None

        self.alpha = 0.5

        # local and remote after partition
        self.delta = 0
        self.config = None

        self.local = 0
        self.remote = 0
        self.remote_deadline = 0
        self.local_to_remote_size = 0
        self.local_only_energy = 0

        self.local_only_enabled = True

        self.total_computation = 0

        # job queue
        self.queue = []

    def rate(self, p, bandwidth, k):
        return bandwidth * math.log2(1 + p * math.pow(self.H[k], 2) / math.pow(10, -9))

    """
       Given network and computation resource from the edge node, select optimal DAG partition
    """
    def select_partition(self, full, epsilon, edge_id, f_e=None, bw=None):
        validation = []
        if full == False:
            for delta in range(len(self.DAG.jobs) - 1):
                self.local, self.remote, data = 0, 0, self.DAG.jobs[delta].output_data
                for m in range(0, delta):
                    self.local += self.DAG.jobs[m].computation
                for m in range(delta, self.DAG.length):
                    self.remote += self.DAG.jobs[m].computation
                if delta == 0:
                    self.local_to_remote_size = self.DAG.jobs[delta].input_data
                else:
                    self.local_to_remote_size = self.DAG.jobs[delta].output_data
                config = self.local_optimal_resource(delta, epsilon, edge_id, f_e, bw)
                if config is not None and (config[1] < self.local_only_energy or not self.local_only_enabled):
                    validation.append(config)
        else:
            delta = 0
            self.local, self.remote = 0, 0
            for m in range(0, self.DAG.length):
                self.remote += self.DAG.jobs[m].computation
            self.local_to_remote_size = self.DAG.jobs[delta].input_data
            config = self.local_optimal_resource(delta, epsilon, edge_id, f_e, bw)
            if config is not None and (config[1] < self.local_only_energy or not self.local_only_enabled):
                validation.append(config)
        if len(validation) > 0:
            validation.sort(key=lambda x: x[1])
            # print("set config", validation)
            # self.delta = validation[0][0]
            return validation[0]
        else:
            #print("run task locally")
            return None

    def local_optimal_resource(self, delta, epsilon, edge_id, f_e, bw, N=math.pow(10, -9)):
        # max transmission time
        TM = self.DAG.D/1000 - self.remote / f_e - self.local / self.freq
        # min transmission time
        TS = self.local_to_remote_size / self.rate(self.p_max/(math.ceil(bw/math.pow(10, 6))), bw, edge_id)
        if TM < TS:
            #print("user", self.task_id, "can not be offloading by delta", delta
            #      , "TS, TM", round(TS, 4), round(TM, 4), self.local_to_remote_size / 8000
            #      , "bw", bw/math.pow(10, 6), "local", self.local/math.pow(10, 9), "remote", self.remote/math.pow(10, 9))
            return None
        #else:
            #print("user", self.task_id, "can be offloading by delta", delta, "TS, TM", TS,
            #      TM)
        t = TS
        tt = []
        ee = []
        dd = []
        config = None
        it = 0
        while t <= TM:
            #  第一介导数
            a = N * self.local_to_remote_size * math.pow(2, self.local_to_remote_size / (bw * t)) * math.log(2) - \
                t * bw * N * math.pow(2, self.local_to_remote_size / (bw * t)) + t * bw * N
            c = math.pow(self.H[edge_id], 2) * t * bw
            directive = 2 * self.k * math.pow(self.local, 3) / math.pow(self.DAG.D/1000 - t - self.remote / f_e, 3) \
                            - (math.ceil(bw/math.pow(10, 6))) * a / c
            # energy
            a = self.k * self.local * math.pow(self.local / (self.DAG.D / 1000 - t - self.remote / f_e), 2)
            b = math.pow(2, self.local_to_remote_size / (bw * t)) - 1
            ee.append(a + math.ceil(bw/math.pow(10, 6)) * t * b * N / math.pow(self.H[edge_id], 2))
            dd.append(directive)
            tt.append(t)
            # print(it, "directive", directive, "t", t)
            """
            if math.fabs(directive) <= 0.0005:
                # minimal energy
                a = self.k * self.local * math.pow(self.local / (self.DAG.D / 1000 - t - self.remote / f_e), 2)
                b = math.pow(2, self.local_to_remote_size / (bw * t)) - 1
                energy = a + math.ceil(bw / math.pow(10, 6)) * t * b * N / math.pow(self.H[edge_id], 2)
                cpu = self.local / (self.DAG.D / 1000 - t - self.remote / f_e)
                power = b * N / math.pow(self.H[edge_id], 2)
                config = [delta, energy, cpu, power, t, directive]

                if self.local != 0:
                    diff = self.DAG.D / 1000 - self.remote / f_e - self.local / cpu - t
                else:
                    diff = self.DAG.D / 1000 - self.remote / f_e - t

                #print("XXXXXXXXX", "user", self.task_id, "diff", diff, self.remote / f_e, t
                #      , "data", t * self.rate(power, bw, edge_id), self.local_to_remote_size
                #      , "time", self.local_to_remote_size / self.rate(power, bw, edge_id),
                #      "directive", directive)
                break
            """
            t = t + TM * epsilon
            # t = t - 0.1 * directive
            it = it + 1

        if config is None:
            sm = 100
            si = -1
            for i in range(len(ee)):
                if si == -1 or ee[i] < sm:
                    sm = ee[i]
                    si = tt[i]
            t = si
            a = self.k * self.local * math.pow(self.local / (self.DAG.D / 1000 - t - self.remote / f_e), 2)
            b = math.pow(2, self.local_to_remote_size / (bw * t)) - 1
            energy = a + math.ceil(bw/math.pow(10, 6)) * t * b * N / math.pow(self.H[edge_id], 2)
            cpu = self.local / (self.DAG.D / 1000 - t - self.remote / f_e)
            power = b * N / math.pow(self.H[edge_id], 2)

            #  第一介导数
            a = N * self.local_to_remote_size * math.pow(2, self.local_to_remote_size / (bw * t)) * math.log(2) - \
                t * bw * N * math.pow(2, self.local_to_remote_size / (bw * t)) + t * bw * N
            c = math.pow(self.H[edge_id], 2) * t * bw
            directive = 2 * self.k * math.pow(self.local, 3) / math.pow(self.DAG.D/1000 - t - self.remote / f_e, 3) \
                            - (math.ceil(bw/math.pow(10, 6))) * a / c

            if self.local != 0:
                diff = self.DAG.D / 1000 - self.remote / f_e - self.local / cpu - t
            else:
                diff = self.DAG.D / 1000 - self.remote / f_e - t

            # print("user", self.task_id, "diff", diff, self.remote / f_e, t
            #     , "data", t * self.rate(power, bw, edge_id), self.local_to_remote_size
            #      , "time", self.local_to_remote_size / self.rate(power, bw, edge_id), "directive", directive)

            config = [delta, energy, cpu, power, t, directive]

        # print(config)
        #plt.plot(tt, ee)
        #plt.plot(tt, dd)
        #plt.xlim([TS, TM])
        #plt.show()
        return config
        # [313.0, 61.0, 452.0, 228.0, 478.0] [0.0004, 0.2796, 0.0497, 0.4252, 0.2154]

    def remote_execution(self, f_e, bw, edge_id):
        if self.config is None:
            return -1, 0

        cpu = self.config[2] * 1.00005
        power = self.config[3] * 1.00005
        delta = self.config[0]

        self.local, self.remote, data = 0, 0, self.DAG.jobs[delta].output_data
        for m in range(0, delta):
            self.local += self.DAG.jobs[m].computation
        for m in range(delta, self.DAG.length):
            self.remote += self.DAG.jobs[m].computation

        if delta == 0:
            self.local_to_remote_size = self.DAG.jobs[delta].input_data
            computation_time = 0
        else:
            self.local_to_remote_size = self.DAG.jobs[delta].output_data
            computation_time = self.local / cpu

        t = self.local_to_remote_size / self.rate(power, bw, edge_id)

        if self.local != 0:
            diff = self.DAG.D / 1000 - self.remote / f_e - self.local / cpu - t
        else:
            diff = self.DAG.D / 1000 - self.remote / f_e - t

        #print("????", "user", self.task_id, "diff", diff, self.remote / f_e, t
        #      , "data", t * self.rate(power, bw, edge_id), self.local_to_remote_size
        #      , "time", self.local_to_remote_size / self.rate(power, bw, edge_id))
        # print(self.task_id, self.local / cpu, cpu/math.pow(10, 9), self.local/math.pow(10, 9), computation_time)
        # minimal energy
        a = self.k * self.local * math.pow(cpu, 2)
        energy = a + t * power * math.ceil(bw/math.pow(10, 6))

        if t + self.remote/f_e + computation_time <= self.DAG.D/1000:
            return energy, 1, t, computation_time, self.remote/f_e
        else:
            # print(self.task_id, ">>>>>>>>>>>>>", t + self.remote/f_e + computation_time - self.DAG.D/1000)
            return energy, 0, t, computation_time, self.remote/f_e

    def local_only_execution(self):
        total_computation = 0
        for m in range(self.DAG.length):
            total_computation += self.DAG.jobs[m].computation
        self.total_computation = total_computation / math.pow(10, 9)
        cpu = 1000 * total_computation / self.DAG.D
        if cpu > self.freq:
            self.local_only_enabled = False
        self.local_only_energy = self.k * total_computation * math.pow(cpu, 2)

    def inital_DAG(self, task_id, config=None, T=None, D=None):
        self.DAG = DAG(task_id, self.freq, T=T, D=D)
        if config is not None:
            self.DAG.create_from_config(config)
        else:
            self.DAG.create()
        self.DAG.get_valid_partition()
        self.DAG.local_only_compute_energy_time(self.freq, self.k)
