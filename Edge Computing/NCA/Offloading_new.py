import math
import numpy as np
import random

class Offloading:

    def __init__(self, W, edge_cpu=20, e=0.001, g=0.95, number_of_user=5, number_of_edge=3,  p_k=math.pow(10, -27)):
        self.step = None
        self.g = g
        self.number_of_user = number_of_user
        self.number_of_edge = number_of_edge
        self.k = p_k
        self.e = e
        # initialize
        self.edge_cpu = np.array([random.randint(edge_cpu, edge_cpu) * math.pow(10, 9) for x in range(self.number_of_edge)])
        self.user_cpu = np.array([random.uniform(0.5, 1.5) * math.pow(10, 9) for x in range(self.number_of_user)])

        self.full_offload = np.zeros(self.number_of_user)
        self.D_n = np.array([random.uniform(0.5, 1) for x in range(self.number_of_user)])
        print("r.D_n=np.array(", list(self.D_n), ")")

        self.Y_n = np.array([random.randint(100, 200) * 8000 * random.randint(20, 70) for n in range(number_of_user)])
        self.X_n = np.array([random.randint(250, 500) * 8000 * random.randint(50, 150) for n in range(self.number_of_user)])

        # 0 = local,  1 = full, 2 = partial
        self.full_offload = np.zeros(self.number_of_user) + 2

        print("r.X_n=np.array(", list(self.X_n), ")")
        print("r.Y_n=np.array(", list(self.Y_n), ")")
        self.f_n = np.array([self.user_cpu[n] for n in range(self.number_of_user)])
        print("r.f_n=np.array(", list(self.f_n), ")")

        self.P_max = np.array([random.uniform(0.5, 1) for n in range(self.number_of_user)])
        print("r.P_max=np.array(", list(self.P_max), ")")

        self.A = np.array([random.randint(750, 1500) * 8000. for n in range(self.number_of_user)])
        self.B = np.array([random.randint(350, 1000) * 8000. for n in range(self.number_of_user)])

        print("r.A=np.array(", list(self.A),")/1.3")
        print("r.B=np.array(", list(self.B),")/1.3")

        self.N_0 = math.pow(10, -9)
        modevalue = np.sqrt(2 / np.pi) * math.pow(10, -3)
        self.H = np.array([[np.random.rayleigh(modevalue) for y in range(number_of_edge)] for x in range(number_of_user)])

        print("r.H=np.array(", list(self.H.reshape(-1)), ")")

        print("r.H=np.array(", list(self.H), ")")

        self.W = W

        self.a_n_k = np.array([[1 for k in range(self.number_of_edge)] for n in range(self.number_of_user)])

        self.c_k = None
        self.c_n_k = None
        self.default_channel = None

        self.l_n = None
        self.var_k = None
        self.v_n = None
        self.d_n_k = None
        self.ct_n_k = None
        self.t_n_k = None
        self.p_n_k = None

        self.f_n_k = None
        self.f_n_r_k = None

        self.delta_l_n = None
        self.delta_var_k = None
        self.delta_d_n_k = None

        self.chs = False
        self.loc_only_e = np.zeros(self.number_of_user)

    def set_initial_sub_channel(self, ch, default_channel, chs=None):
        # number of channel
        self.c_k = [ch for k in range(self.number_of_edge)]
        # initial channel
        self.c_n_k = [[[0 for c in range(self.c_k[k])]
                       for k in range(self.number_of_edge)]
                      for n in range(self.number_of_user)]

        if chs is not None:
            for k in range(self.number_of_edge):
                c = 0
                for n in range(self.number_of_user):
                    rep = 0
                    while rep < chs[n]:
                        self.c_n_k[n][k][c] = 1
                        c += 1
                        rep += 1
            self.chs = True
        else:
            self.default_channel = default_channel
            for k in range(self.number_of_edge):
                n = 0
                c = 0
                while c < self.c_k[k] and n < self.number_of_user:
                    rep = 0
                    while rep < self.default_channel:
                        self.c_n_k[n][k][c] = 1
                        c += 1
                        rep += 1
                    n += 1

    def set_multipliers(self, step=0.00001, p_adjust=1.5, v_n=0.1, var_k=math.pow(10, -11),
                        delta_l_n=math.pow(10, -12), delta_var_k=0.3 * math.pow(10, -19), delta_d_n_k=math.pow(10, -14)):
        self.step = step
        self.d_n_k = np.array([[math.pow(self.f_n[n], 3) * 2 * self.k
                                for k in range(self.number_of_edge)] for n in range(self.number_of_user)])
        self.v_n = np.zeros(self.number_of_user) + v_n
        self.l_n = [(self.get_ch_number(n) * self.P_max[n] / p_adjust + self.d_n_k[n][0])
                      /(self.get_ch_number(n) * self.W * math.log2(1 + (self.P_max[n] / p_adjust ) * math.pow(self.H[n][0], 2)/self.N_0))
                    for n in range(self.number_of_user)]
        self.delta_l_n = delta_l_n
        self.delta_d_n_k = delta_d_n_k

    def set_initial_values(self):
        self.f_n_k = np.array([[0.0 for k in range(self.number_of_edge)] for n in range(self.number_of_user)])
        # 0 = local,  1 = full, 2 = partial
        for k in range(self.number_of_edge):
            for n in range(self.number_of_user):
                if self.full_offload[n] == 2:
                    self.f_n_k[n][k] = self.get_f_n_k(n, k)
        self.f_n_r_k = np.array([[self.get_f_n_r_k(n, k) for k in range(self.number_of_edge)] for n in range(self.number_of_user)])
        self.p_n_k = np.array([[self.get_p_n_k(n, k) for k in range(self.number_of_edge)] for n in range(self.number_of_user)])
        self.t_n_k = np.array([[self.get_t(n, k) for k in range(self.number_of_edge)] for n in range(self.number_of_user)])

        print("P_max", self.P_max.reshape(-1))
        print("p_n_k", self.p_n_k.reshape(-1))
        print("f_n", self.f_n.reshape(-1)/math.pow(10, 9))
        print("f_n_k", self.f_n_k.reshape(-1)/math.pow(10, 9))
        print("deadline", self.D_n.reshape(-1))
        print("transmission time", self.t_n_k.reshape(-1))
        print("transmission power", self.p_n_k.reshape(-1))

        chs = []
        for n in range(self.number_of_user):
            chs.append(self.get_ch_number(n))
        print("channel", list(chs))

    def get_f_n_k(self, n, k):
        a = self.d_n_k[n][k]
        b = 2 * self.k
        return min(self.f_n[n], (a/b)**(1. / 3))

    def get_f_n_r_k(self, tn, k):
        f_sum = 0
        for n in range(self.number_of_user):
            f_sum += math.sqrt(self.d_n_k[n][k] * self.X_n[n])
        return self.edge_cpu[k] * math.sqrt(self.d_n_k[tn][k] * self.X_n[tn]) / f_sum

    """
    do bisection to obtain the optimal transmission power
    """
    def get_p_n_k(self, n, k):
        ch = self.get_ch_number(n)
        G = math.pow(self.H[n][k], 2) / self.N_0
        station_p = (math.pow(self.H[n][k], 2) * self.W * self.l_n[n] - self.N_0 * math.log(2)) / (math.pow(self.H[n][k], 2) * math.log(2))
        min_value = ch * station_p + self.d_n_k[n][k] - ch * self.l_n[n] * self.W * math.log2(1 + G * station_p)
        if n == 33:
            print(station_p, self.P_max[n], "min_value", min_value, "ch", ch)
            print(self.H[n][k], self.l_n[n], self.d_n_k[n][k])

        FLAG = 0
        if station_p >= self.P_max[n]:
            h = self.P_max[n]
            l = 0
            FLAG = 1
        else:
            if ch * self.P_max[n] + self.d_n_k[n][k] - ch * self.l_n[n] * self.W * math.log2(1 + G * self.P_max[n]) >=0:
                h = self.P_max[n]
                l = station_p
                FLAG = 2
            else:
                h = station_p
                l = 0
                FLAG = 3

        target = ch * (h+l)/2 + self.d_n_k[n][k] - self.l_n[n] * ch * self.W * math.log2(1 + G * (h + l)/2)
        while math.fabs(target) >= 0.00001:
            #if n == 13:
                #print(target, (h + l) / 2, h, l)
            if target < 0:
                if FLAG == 2:
                    l = (h + l) / 2
                if FLAG == 3:
                    h = (h + l) / 2
                if FLAG == 1:
                    h = (h + l) / 2
            else:
                if FLAG == 2:
                    h = (h + l) / 2
                if FLAG == 3:
                    l = (h + l) / 2
                if FLAG == 1:
                    l = (h + l) / 2

            target = ch * (h + l) / 2 + self.d_n_k[n][k] - self.l_n[n] * ch * self.W * math.log2(1 + G * (h + l) / 2)
        #print(n, ">>>>>", target, (h+l)/2)
        return (h + l) / 2

    def get_t(self, n, k):
        a = math.pow(self.H[n][k], 2) * self.W
        b = self.N_0 * math.log(2) + math.pow(self.H[n][k], 2) * self.p_n_k[n][k] * math.log(2)
        #print((self.l_n[n] * (a/b) - 1))
        r = 0
        for c in range(self.c_k[k]):
            r += self.c_n_k[n][k][c] * self.W * math.log2(
                1 + self.p_n_k[n][k] * math.pow(self.H[n][k], 2) / self.N_0)

        if self.full_offload[n] == 2:
            return self.B[n]/r
        else:
            return self.A[n]/r
        #return self.v_n[n] / (self.l_n[n] * (a/b) - 1)

    def update(self, run, t, verbose=False):
        stop = True
        diff_data = 0
        for n in range(self.number_of_user):
            if self.full_offload[n] == 0:
                continue
            diff = 0
            for k in range(self.number_of_edge):
                r = 0
                for c in range(self.c_k[k]):
                    r += self.c_n_k[n][k][c] * self.W * math.log2(
                        1 + self.p_n_k[n][k] * math.pow(self.H[n][k], 2) / self.N_0)
                if self.full_offload[n] == 2:
                    diff += (self.B[n] - self.t_n_k[n][k] * r)
                else:
                    diff += (self.A[n] - self.t_n_k[n][k] * r)
            new_l_n = max(0, self.l_n[n] - self.delta_l_n * math.sqrt(self.step / t) * diff)
            diff_data += diff
            if abs(diff) > math.pow(10, -5):
                stop = False
            self.l_n[n] = new_l_n

        for n in range(self.number_of_user):
            if self.full_offload[n] == 0:
                continue
            diff = 0.
            for k in range(self.number_of_edge):
                for c in range(self.c_k[k]):
                    if self.c_n_k[n][k][c] == 1:
                        e = self.p_n_k[n][k]
                        diff += e
            diff = diff - self.P_max[n]
            new_v_n = max(0, self.v_n[n] - math.sqrt(self.step / t) * diff)
            if abs(diff) > math.pow(10, -5):
                stop = False
            self.v_n[n] = new_v_n

        diff_compute = 0
        for n in range(self.number_of_user):
            if self.full_offload[n] != 2:
                continue
            for k in range(self.number_of_edge):
                diff = self.t_n_k[n][k] + self.Y_n[n]/self.f_n_k[n][k] + self.X_n[n]/self.f_n_r_k[n][k] - self.D_n[n]
                new_d_n_k = max(0, self.d_n_k[n][k] - self.delta_d_n_k * math.sqrt(self.step / t) * diff)
                if abs(diff) > math.pow(10, -5):
                    stop = False
                self.d_n_k[n][k] = new_d_n_k
                diff_compute += diff

        print(run, t, ", data", diff_data/8000, "compute", diff_compute/math.pow(10, 9)
                  , round(self.p_n_k[0][0], 5), round(self.t_n_k[0][0],5)
                  , "l_n", self.l_n[0])
                  #, self.d_n_k[4][k], self.ct_n_k[4][k], self.f_n_k[4][k]/math.pow(10, 9))
                  # , "n13", round(e, 5), round(1 - e/self.loc_only_e[12], 5))

        return stop

    def checkpoint(self, t, check):
        if t % 1500 * check == 0:
            k = 0
            for n in range(self.number_of_user):
                b = self.get_ch_number(n) * self.p_n_k[n][k] * self.t_n_k[n][k]
                if self.full_offload[n] == 2:
                    c = self.Y_n[n] * self.k * math.pow(self.f_n_k[n][k], 2)
                else:
                    c = 0
                e = b + c
                if 1 - e/self.loc_only_e[n] <= 0.25:
                    if self.full_offload[n] == 1:
                        self.full_offload[n] = 0
                        for k in range(self.number_of_edge):
                            for c in range(self.c_k[k]):
                                self.c_n_k[n][k][c] = 0
                    else:
                        self.full_offload[n] = 1
            return check + 1
        return check

    def get_ch_number(self, n):
        total = 0
        for k in range(self.number_of_edge):
            for c in range(self.c_k[k]):
                if self.c_n_k[n][k][c] == 1:
                    total += 1
        return total

    def new_value(self):
        for k in range(self.number_of_edge):
            for n in range(self.number_of_user):
                if self.full_offload[n] == 0:
                    continue
                a = self.get_p_n_k(n, k)
                # update transmission power
                if a < 0:
                    print(n, ">>>>>>>>>>>>>>>>>>", self.v_n[n], "self.l_n[n]", self.l_n[n], self.N_0 / math.pow(self.H[n][k],2))
                self.p_n_k[n][k] = min(max(0, a), self.P_max[n]/self.get_ch_number(n))
                if self.full_offload[n] == 2:
                    self.f_n_k[n][k] = self.get_f_n_k(n, k)
                self.t_n_k[n][k] = self.get_t(n, k)

    def assign_ch(self, t):
        if self.chs:
            return

        for k in range(self.number_of_edge):
            for c in range(self.number_of_user * self.default_channel, self.c_k[k]):
                opt = -1
                max_gain = 999
                for n in range(self.number_of_user):
                    if self.full_offload[n] == 0:
                        continue
                    if self.full_offload[n] == 2:
                        need_rate = self.B[n] / self.t_n_k[n][k]
                    else:
                        need_rate = self.A[n] / self.t_n_k[n][k]

                    cgs = self.get_ch_number(n)
                    a = need_rate / (self.W * (cgs + 1))
                    b = need_rate / (self.W * cgs)
                    p_new = self.t_n_k[n][k] * (cgs + 1) * (self.N_0 / math.pow(self.H[n][k], 2)) * (math.pow(2, a) - 1)
                    p_old = self.t_n_k[n][k] * cgs * (self.N_0 / math.pow(self.H[n][k], 2)) * (math.pow(2, b) - 1)
                    gain = p_old - p_new

                    if gain > max_gain or opt == -1:
                        opt = n
                        max_gain = gain

                if opt != -1:
                    for n in range(self.number_of_user):
                        if n == opt and self.full_offload[n] > 0:
                            self.c_n_k[n][k][c] = 1
                        else:
                            self.c_n_k[n][k][c] = 0
        if t % 300 == 0:
            chs = []
            for n in range(self.number_of_user):
                chs.append(self.get_ch_number(n))
            print("channel", list(chs))

    def run(self, run, t=1, t_max=5000, t_delay=1500, t_stable=2000):
        check = 1
        while t <= t_max:
            stop1 = self.update(run, t)
            t = t + 1
            #if t <= (t_max - t_stable) and (t >= t_delay):
                #self.assign_ch(t)
            self.new_value()
            #if t >= t_delay + 1000:
            #    check = self.checkpoint(t, check)
            if stop1:
                break
        #print(list(self.v_n))
