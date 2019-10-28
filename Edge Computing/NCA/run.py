import math
import numpy as np


def test(r):

    finish = []
    ee_l = []
    for n in range(r.number_of_user):
        f_opt = min((r.X_n[n] + r.Y_n[n]) / r.D_n[n], r.user_cpu[n])
        e_n = r.g * r.k * (r.X_n[n] + r.Y_n[n]) * math.pow(f_opt, 2)
        ee_l.append(e_n)
        finish.append((r.X_n[n] + r.Y_n[n])/f_opt - r.D_n[n])
        r.loc_only_e[n] = e_n

    ee = []
    ee_c = []
    ee_t = []
    t_e = []
    t_p = np.zeros(r.number_of_user)
    finish_time = np.zeros(r.number_of_user)
    t_local_computation = np.zeros(r.number_of_user)
    ee_p = []
    b = np.zeros(r.number_of_user)
    f_cpu = np.zeros(r.number_of_user)
    f = 0

    channels = []
    transmission_power_total = np.array([[ r.p_n[n] * r.get_ch_number(n)
                                         for k in range(r.number_of_edge)] for n in range(r.number_of_user)])
    for n in range(r.number_of_user):
        for k in range(r.number_of_edge):
            if r.full_offload[n] > 0:
                chs = r.get_ch_number(n)
                channels.append(chs)
                rate = r.W * chs * math.log2(1 + r.p_n[n] * math.pow(r.H[n][k], 2) / r.N_0)

                if r.full_offload[n] == 1:
                    r.X_n[n] = r.X_n[n] + r.Y_n[n]
                    r.Y_n[n] = 0.0
                    r.B[n] = r.A[n]
                    allocated_f = r.f_n_k[n][k]
                    t_e.append(r.X_n[n] / allocated_f)
                    t_local_computation[n] = 0
                    finish_time[n] = r.B[n]/rate + r.X_n[n] / allocated_f
                else:
                    allocated_f = r.f_n_k[n][k]
                    t_e.append(r.X_n[n] / allocated_f)
                    t_local_computation[n] = r.Y_n[n]/r.f_n[n]
                    finish_time[n] = r.B[n]/rate + r.X_n[n] / allocated_f + r.Y_n[n]/r.f_n[n]

                e_n_k = r.p_n[n] * chs * (r.B[n] / rate) + r.k * r.Y_n[n] * math.pow(r.f_n[n], 2)
                b[n] = r.t_n[n] * rate
                ee.append(e_n_k)
                ee_c.append(r.k * r.Y_n[n] * math.pow(r.f_n[n], 2))

                ee_t.append(r.p_n[n] * chs * (r.B[n] / rate))
                pp = r.P_max[n] / chs
                rate = chs * r.W * math.log2(1 + pp * math.pow(r.H[n][k], 2) / r.N_0)
                t_max = r.B[n] / rate
                t_p[n] = t_max
                ee_p.append(r.P_max[n] * t_max)
                f += r.f_n_k[n][k]
                f_cpu[n] = r.f_n_k[n][k]
            else:
                ee.append(ee_l[n])
                ee_t.append(0)
                ee_c.append(0)
                t_e.append(0)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>time>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("task          deadline", list(np.round(r.D_n, 4).reshape(-1)))
    print("finish            time", list(np.round(finish_time, 4)))
    print("transmission      time", list(np.round(r.t_n, 4).reshape(-1)))
    print("edge  computation time", list(np.round(t_e, 4).reshape(-1)))
    print("local computation time", list(np.round(t_local_computation, 4).reshape(-1)))
    print("local only        time", list(np.round(finish, 4)))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>CPU and Power>>>>>>>>>>>>>>>>>>>>>")
    print("      edge CPU used", f / math.pow(10, 9), "max", r.edge_cpu[k] / math.pow(10, 9))
    print("device    max power", list(np.round(r.P_max, 4)))
    print("transmission  power", list(np.round(transmission_power_total, 4).reshape(-1)))
    print("edge  allocated CPU", list(np.round(f_cpu / math.pow(10, 9), 4)))
    print("local allocated CPU", list(np.round(r.f_n / math.pow(10, 9), 4).reshape(-1)))
    print("max local       CPU", list(np.round(r.user_cpu / math.pow(10, 9), 4)))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("transmission           gain", list(np.round(r.H, 4).reshape(-1)))
    print("check >>>          channels", list(channels))
    print("check >>> transmission data", list(np.array(r.B / 8000).astype(int)))
    print("check >>> transmission data", list(np.array(b / 8000).astype(int)))
    print("check >>> local computation data", list(np.round(r.Y_n / math.pow(10, 9), 5)))
    print("check >>> ede  computation data", list(np.round(r.X_n / math.pow(10, 9), 5)))
    print("check >>> local computation data", list(np.round(r.f_n * t_local_computation / math.pow(10, 9), 5).reshape(-1)))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>Energy>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("           offloading E", list(np.round(ee, 5)))
    print("  offloading transmit E", list(np.round(ee_t, 5)))
    print(" local      computing E", list(np.round(ee_c, 5)))
    print("           local only E", list(np.round(ee_l, 5)))
    print("   Total Energy offload", np.sum(ee))
    # print("Total Energy via task offloading using max P", np.sum(ee_p))
    print("Total Energy Local Only", np.sum(ee_l))
    print("Saving", 1 - np.array(ee) / np.array(ee_l))
    print("Saving", 1 - np.sum(ee) / np.sum(ee_l))
    # print(r.v_n)
    return np.sum(ee), np.sum(ee_l), 1 - np.sum(ee) / np.sum(ee_l)
