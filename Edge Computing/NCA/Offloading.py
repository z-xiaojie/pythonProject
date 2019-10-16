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

        self.Y_n = np.array([random.randint(100, 550) * 8000 * random.randint(50, 70) for n in range(number_of_user)])
        self.X_n = np.array([random.randint(250, 1000) * 8000 * random.randint(50, 150) for n in range(self.number_of_user)])

        # 0 = local,  1 = full, 2 = partial
        self.full_offload = np.zeros(self.number_of_user) + 2

        print("r.X_n=np.array(", list(self.X_n), ")")
        print("r.Y_n=np.array(", list(self.Y_n), ")")
        self.f_n = np.array([self.user_cpu[n] for n in range(self.number_of_user)])
        print("r.f_n=np.array(", list(self.f_n), ")")

        self.P_max = np.array([random.uniform(0.5, 1) for n in range(self.number_of_user)])
        print("r.P_max=np.array(", list(self.P_max), ")")

        self.A = np.array([random.randint(750, 1500) * 8000. for n in range(self.number_of_user)])/1.3
        self.B = np.array([random.randint(350, 1000) * 8000. for n in range(self.number_of_user)])/1.3

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

        self.delta_l_n = None
        self.delta_var_k = None
        self.delta_d_n_k = None

        self.chs = False
        self.loc_only_e = np.zeros(self.number_of_user)

        self.minimal = 0

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
        self.v_n = np.zeros(self.number_of_user) + v_n
        self.l_n = [(self.N_0 / math.pow(np.max(self.H[n]), 2) + self.P_max[n]/p_adjust) * (1 + self.v_n[n]) * math.log(2) / self.W
                    for n in range(self.number_of_user)]
        self.p_n_k = np.array([[
            #max(0, self.l_n[n] * self.W / ((1 + self.v_n[n]) * math.log(2)) - self.N_0 / math.pow(self.H[n][k], 2))
            self.P_max[n] / p_adjust
            for k in range(self.number_of_edge)]
            for n in range(self.number_of_user)])
        self.var_k = np.zeros(self.number_of_user) + var_k
        self.d_n_k = np.array([[(self.v_n[n] * self.P_max[n] + self.l_n[n] * self.get_gain(n, k)) / self.f_n[n]
                                for k in range(self.number_of_edge)] for n in range(self.number_of_user)])
        self.delta_l_n = delta_l_n
        self.delta_var_k = delta_var_k
        self.delta_d_n_k = delta_d_n_k

    def set_initial_values(self):
        self.f_n_k = np.array([[0.0 for k in range(self.number_of_edge)] for n in range(self.number_of_user)])
        self.ct_n_k = np.array([[0.0 for k in range(self.number_of_edge)] for n in range(self.number_of_user)])
        # 0 = local,  1 = full, 2 = partial
        for k in range(self.number_of_edge):
            for n in range(self.number_of_user):
                if self.full_offload[n] == 2:
                    self.f_n_k[n][k] = self.f_n[n] #self.get_f_n_k(n, k)
                    self.ct_n_k[n][k] = self.get_c_t_n_k(n, k)
        self.t_n_k = np.array([[self.get_t(n,k) for k in range(self.number_of_edge)] for n in range(self.number_of_user)])

        print("P_max", self.P_max.reshape(-1))
        print("p_n_k", self.p_n_k.reshape(-1))
        print("f_n", self.f_n.reshape(-1)/math.pow(10, 9))
        print("f_n_k", self.f_n_k.reshape(-1)/math.pow(10, 9))
        print("deadline", self.D_n.reshape(-1))
        print("computation time", self.ct_n_k.reshape(-1))
        print("transmission time", self.t_n_k.reshape(-1))
        print("transmission power", self.p_n_k.reshape(-1))

        chs = []
        for n in range(self.number_of_user):
            chs.append(self.get_ch_number(n))
        print("channel", list(chs))

    def get_gain(self, n , k):
        sum = 0
        for c in range(self.c_k[k]):
            G = self.p_n_k[n][k] * math.pow(self.H[n][k], 2) / self.N_0
            r_n_k_c = self.W * math.log2(1 + G)
            a = math.log(2) * r_n_k_c * (1 + G) - self.W * G
            b = math.log(2) * (1 + G)
            sum += self.c_n_k[n][k][c] * (a / b)
        return sum

    def get_c_t_n_k(self, n, k):
        a = self.v_n[n] * self.P_max[n] + self.l_n[n] * self.get_gain(n, k)
        b = math.pow(self.d_n_k[n][k], 2)
        c = 2 * self.k * self.Y_n[n] * a / b
        # print(">>>>>>>>", a, 2 * self.f_n[n] * self.k * self.Y_n[n] / self.d_n_k[n][k])
        return min(max(0., c), 2 * self.f_n[n] * self.k * self.Y_n[n] / self.d_n_k[n][k])
        # return max(2 * self.k * self.Y_n[n] * self.f_n_k[n][k] / self.d_n_k[n][k], self.Y_n[n] / self.f_n[n])

    def get_f_n_k(self, n, k):
        a = self.v_n[n] * self.P_max[n] + self.l_n[n] * self.get_gain(n,k)
        b = self.d_n_k[n][k]
        return min(self.f_n[n], (a/b))

    #def show_minial_channel(self):
        #for n in range(self.number_of_user):
            #r_max =

    def get_t(self, n, k):

        if self.full_offload[n] == 2:
            b = self.var_k[k] * self.X_n[n]
        else:
            b = self.var_k[k] * (self.X_n[n] + self.Y_n[n])

        c = self.v_n[n] * self.P_max[n] + self.l_n[n] * self.get_gain(n,k)
        if c == 0:
            print(n, "XXXXXXXXXX", self.l_n[n], self.get_gain(n, k), self.p_n_k[n][k])
        d = math.sqrt(b / c)

        r_max = self.W * self.get_ch_number(n) * math.log2(1 + (self.P_max[n]/self.get_ch_number(n)) * math.pow(self.H[n][k], 2) / self.N_0)

        #if self.D_n[n] - d - self.ct_n_k[n][k] <= 0:
            #print(n, self.var_k[k], self.get_min_var(k), self.D_n[n], d, self.ct_n_k[n][k])
            #self.var_k[k] = self.get_min_var(k)/1.01
            #return self.t_n_k[n][k]

        if self.full_offload[n] == 2:
            t_min = self.B[n] / r_max
        else:
            t_min = self.A[n] / r_max

        #if n == 5:
            #print(t_min, self.D_n[n] - d - self.ct_n_k[n][k])
        # print("t_min", t_min)
        return max(t_min, self.D_n[n] - d - self.ct_n_k[n][k])
        #return self.D_n[n] - d - self.ct_n_k[n][k]

    def calculate_energy(self):
        e_n_k = 0
        for n in range(self.number_of_user):
            for k in range(self.number_of_edge):
                if self.full_offload[n] == 2:
                    e_n_k += self.p_n_k[n][k] * self.get_ch_number(n) * self.t_n_k[n][k] \
                            + self.k * self.Y_n[n] * math.pow(self.f_n_k[n][k], 2)
                elif self.full_offload[n] == 1:
                    e_n_k += self.p_n_k[n][k] * self.get_ch_number(n) * self.t_n_k[n][k]
        return e_n_k

    def get_min_var(self, k):
        v = np.zeros(self.number_of_user)
        for n in range(self.number_of_user):
            if self.full_offload[n] == 2:
                a = self.X_n[n]
            else:
                a = self.X_n[n] + self.Y_n[n]

            v[n] = math.pow(self.D_n[n] - self.ct_n_k[n][k], 2) * (
                    self.v_n[n] * self.P_max[n] + self.l_n[n] * self.get_gain(n, k)) / a
        v_min = min(v)
        return v_min

    def update(self, run, t, ttt, verbose=False):
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
            if n == 155:
                print(">>>>>>diff ", diff, self.l_n[n] + self.delta_l_n * math.sqrt(self.step / t) * diff)

            if self.full_offload[n] == 2:
                tt = self.D_n[n] - self.X_n[n]/self.edge_cpu[0] - self.Y_n[n]/self.f_n[n]
                rate_min = self.B[n] / tt
                p_min = (math.pow(2, rate_min / (self.W * self.get_ch_number(n))) - 1) * self.N_0 / math.pow(self.H[n][0], 2)
            else:
                tt = self.D_n[n] - self.X_n[n] / self.edge_cpu[0] - self.Y_n[n] / self.f_n[n]
                rate_min = self.A[n] / tt
                p_min = (math.pow(2, rate_min / (self.W * self.get_ch_number(n))) - 1) * self.N_0 / math.pow(
                    self.H[n][0], 2)
            # (1 + self.v_n[n]) * math.log(2) * (p_min + self.N_0 / math.pow(self.H[n][0], 2)) / self.W
            new_l_n = max(0
                          , self.l_n[n] + self.delta_l_n * math.sqrt(self.step / t) * diff)

            diff_data += diff
            if abs(diff/8000) > math.pow(10, -5):
                stop = False
            self.l_n[n] = new_l_n

        for n in range(self.number_of_user):
            if self.full_offload[n] == 0:
                continue
            diff = 0.
            for k in range(self.number_of_edge):
                for c in range(self.c_k[k]):
                    if self.c_n_k[n][k][c] == 1:
                        e = self.t_n_k[n][k] * self.p_n_k[n][k]
                        diff += e
            diff = diff - self.t_n_k[n][k] * self.P_max[n]
            new_v_n = max(0, self.v_n[n] + math.sqrt(self.step / t) * diff)
            #if abs(diff) > math.pow(10, -5):
               # stop = False
            self.v_n[n] = new_v_n
           # if self.get_ch_number(n) * self.p_n_k[n][0] < self.P_max[n]:
             #   self.v_n[n] = 0

        diff_compute = 0
        for n in range(self.number_of_user):
            if self.full_offload[n] != 2:
                continue
            for k in range(self.number_of_edge):
                diff = self.Y_n[n] - self.ct_n_k[n][k] * self.f_n_k[n][k]
                # (self.v_n[n] * self.P_max[n] + self.l_n[n] * self.get_gain(n, k)) * self.D_n[n] / self.Y_n[n]
                new_d_n_k = max(0, self.d_n_k[n][k] - self.delta_d_n_k * math.sqrt(self.step / t) * diff)
                if n == 222:
                    print("diff", diff, self.d_n_k[n][k] + self.delta_d_n_k * math.sqrt(self.step / t) * diff,
                          self.d_n_k[n][k], self.f_n_k[n][k], self.ct_n_k[n][k])
                if abs(diff/math.pow(10, 9)) > math.pow(10, -5):
                    stop = False
                self.d_n_k[n][k] = new_d_n_k
                diff_compute += diff

        for k in range(self.number_of_edge):
            diff = 0
            Flag = False
            for n in range(self.number_of_user):
                if self.full_offload[n] > 0:
                    if self.full_offload[n] == 2:
                        a = self.X_n[n]
                    else:
                        a = self.X_n[n] + self.Y_n[n]
                    b = self.D_n[n] - self.t_n_k[n][k] - self.ct_n_k[n][k]
                    if b <= 0:
                        if self.full_offload[n] == 2:
                            dd = self.var_k[k] * self.X_n[n]
                        else:
                            dd = self.var_k[k] * (self.X_n[n] + self.Y_n[n])
                        c = self.v_n[n] * self.P_max[n] + self.l_n[n] * self.get_gain(n, k)
                        print("min_c_t_n_k",  self.Y_n[n] / self.f_n[n], "c", c, "dd", dd)
                        print(">>>>>>>>>", n, self.var_k[k], self.D_n[n], "transmission time", self.t_n_k[n][k], self.ct_n_k[n][k], math.sqrt(dd / c))
                        print("data", diff_data/8000)
                        Flag = True
                        break
                    diff += (a / b)
            if Flag:
                break
            diff = diff - self.edge_cpu[k]
            new_var_k = max(0, self.var_k[k] + self.delta_var_k * math.sqrt(self.step / t) * diff)

            """
            b = self.get_ch_number(12) * self.p_n_k[12][k] * self.t_n_k[12][k]
            if self.full_offload[12] == 2:
                c = self.Y_n[12] * self.k * math.pow(self.f_n_k[12][k], 2)
            else:
                c = 0
            e = b + c
            """
            print(run, ttt, "CPU", diff/math.pow(10, 9), ", data", diff_data/8000
                  ,"compute", diff_compute/math.pow(10, 9)
                  #, self.full_offload)
                  #, "var_k", new_var_k, v_min)
                  , self.d_n_k[4][k], self.ct_n_k[4][k], self.l_n[4])
                  #, "n13", round(e, 5), round(1 - e/self.loc_only_e[12], 5))
            if abs(diff/math.pow(10, 9)) > math.pow(10, -5):
                stop = False
            self.var_k[k] = new_var_k  # min(new_var_k, self.get_min_var(k))

        return stop

    def checkpoint(self, t, check):
        if t % 1000 == 0:
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
                a = self.l_n[n] * self.W / ((1 + self.v_n[n]) * math.log(2)) - self.N_0 / math.pow(self.H[n][k],2)
                # update transmission power
                if a < 0:
                    print(n, ">>>>>>>>>>>>>>>>>>", self.v_n[n], "self.l_n[n]", self.l_n[n], self.N_0 / math.pow(self.H[n][k],2))
                self.p_n_k[n][k] = min(max(0, a), self.P_max[n]/self.get_ch_number(n))
                if self.full_offload[n] == 2:
                    self.f_n_k[n][k] = self.get_f_n_k(n, k)
                    self.ct_n_k[n][k] = self.get_c_t_n_k(n, k)
                self.t_n_k[n][k] = self.get_t(n, k)

    def assign_ch(self, t):
        if self.chs:
            return
        """
        self.c_n_k = [[[0 for c in range(self.c_k[k])]
                       for k in range(self.number_of_edge)]
                      for n in range(self.number_of_user)]

        self.minimal = 0
        for k in range(self.number_of_edge):
            c = 0
            for n in range(self.number_of_user):
                t = self.D_n[n] - self.ct_n_k[n][k] - self.t_n_k[n][k]
                if self.full_offload[n] == 2:
                    need_rate = self.B[n] / self.t_n_k[n][k]
                else:
                    need_rate = self.A[n] / self.t_n_k[n][k]
                b = 1
                while b * self.W * math.log2(1 + self.P_max[n]*math.pow(self.H[n][k], 2)/(b*self.N_0)) < need_rate:
                    b = b + 1
                    self.c_n_k[n][k][c] = 1
                    c += 1
                self.minimal += b
        """
        for k in range(self.number_of_edge):
            if self.c_k[k] <= self.minimal:
                continue
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
        if t % 1 == 0:
            chs = []
            for n in range(self.number_of_user):
                chs.append(self.get_ch_number(n))
            print("channel", list(chs))

    def run(self, run, t=1, t_max=5000, t_delay=1500, t_stable=2000):
        check = 1
        ee = []
        while t <= t_max:
            stop1 = self.update(run, 1, t)
            t = t + 1
            if t <= (t_max - t_stable -100) and (t >= t_delay):
                self.assign_ch(t)
            self.new_value()
            # if t <= (t_max - t_stable) and t >= t_delay + 1000:
               # check = self.checkpoint(t, check)
            if t % 100 == 0:
                ee.append(self.calculate_energy())
            if stop1:
                break
            #print(list(np.array(self.ct_n_k).reshape(-1)))
        print(list(self.v_n))
        return ee
