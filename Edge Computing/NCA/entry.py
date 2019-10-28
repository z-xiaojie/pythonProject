import math
import numpy as np
import random
import copy
from Offloading_Mobihoc import Offloading
from run import test
import matplotlib.pyplot as plt


run = 0
total_run = 10
ec_o_avg = np.zeros(total_run)
ec_o_opt = np.zeros(total_run)
ec_l = np.zeros(total_run)
ec_i = np.zeros(total_run)
chs = 16

# create environment
r = Offloading(W=2 * math.pow(10, 6), edge_cpu=8.5, e=math.pow(10, -9), g=1, number_of_user=8, number_of_edge=1)
r.full_offload = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
for n in range(r.number_of_user):
    f_opt = min((r.X_n[n] + r.Y_n[n]) / r.D_n[n], r.f_n[n])
    e_n = r.g * r.k * (r.X_n[n] + r.Y_n[n]) * math.pow(f_opt, 2)
    r.loc_only_e[n] = e_n

while run < total_run:
    r1 = copy.deepcopy(r)
    r2 = copy.deepcopy(r)
    # r1 : opt sub-channel allocation
    r1.set_initial_sub_channel(chs, 1, chs=None)
    if chs <= 24:
        r1.set_multipliers(step=0.001, p_adjust=0.95, delta_l_n=1, delta_d_n=1)
    else:
        r1.set_multipliers(step=0.001 / (2 * chs), p_adjust=0.85, delta_l_n=1, delta_d_n=1)
    r1.set_initial_values()
    ee = r1.run(run, t=1, intervel=50, stop_point=0.0005)
    ec_o_opt[run], ec_l[run], ec_i[run] = test(r1)

    # r2 : average sub-channel allocation
    r2.set_initial_sub_channel(chs, int(chs / 8), chs=None)
    if chs <= 24:
        r2.set_multipliers(step=0.001, p_adjust=0.95, delta_l_n=1, delta_d_n=1)
    else:
        r2.set_multipliers(step=0.001 / (2 * chs), p_adjust=0.85, delta_l_n=1, delta_d_n=1)
    r2.set_initial_values()
    ee = r2.run(run, t=1, intervel=50, stop_point=0.0005)
    ec_o_avg[run], _, _ = test(r2)

    run = run + 1
    chs += 8

print("ec_o_opt=", list(ec_o_opt))
print("ec_o_avg=", list(ec_o_avg))
print("l=", list(ec_l))
plt.plot(ec_o_opt, label="edge_computing_opt")
plt.plot(ec_o_avg, label="edge_computing_avg")
plt.plot(ec_l, label="local_computing")
plt.legend()
plt.show()