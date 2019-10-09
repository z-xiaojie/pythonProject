import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Compare import compare
from Main import create_edge
from User import User
import copy
from Optimization import set_unit_migration_overhead

########################
#   create users
#   transmission time should be controlled in 0 - 1000 ms
#   given bandwidth 40 MHz,  SNR = 0.75 , max_rate = 40 * 0.75 = 30 Mbps
#   if average adjust 0.7, valid rate = 21 Mbps, if 4 users each edge, then real rate = 21 / 4 = 5 Mpbs
#   the data size should be limited in 1 ~ 5 MB
#   execution time are also limited in 300 ~ 1000 ms
########################
def create_users(number_of_user, cur_time=0):
    users = list()
    param = np.array([
    ])
    for n in range(number_of_user):
        #users.append(User(cur_time, n, param[n][0], param[n][1], param[n][2], param[n][3]))
        T = np.random.randint(5, 8)
        users.append(User(cur_time, n, round(np.random.uniform(0.5, 1), 2), np.random.randint(3, 6),
                          T, T))
        users[-1].simple_summary()
    return users


########################
#  entire network data
########################
def load_network_evn():
    raw_data = []
    ch_state = pd.read_csv("train_state23.csv", header=None)
    for i in range(len(ch_state.columns)):
        raw_data.append(ch_state[ch_state.columns[i]])
    return np.array(raw_data)


########################
#  performance compare
#######################
evn = load_network_evn()
number_of_user = 30
number_of_edge = 10
"""
tf = 10
rf = 10
t = a * 10 + (1-a)*10
"""
alpha =  [0, 0.1 , 0.2, 0.3,0.4, 0.5, 0.6, 0.7 ,0.8, 0.9, 1]
change = [1, 1, 1 , 1  , 1  ,  1, 1,  1,1,1,1,1,1,1]
marker = ["+", "1", "2", "3", "4", "+", "x", "|", "*"]

start_time = int(80000 * 1.8)
test = [[0.1, 1], [0.7, 2]]
number_of_run = 50
for i in range(2):
    run = number_of_run
    """
    complete_history = []
    migration_overhead_history = []
    avg_cost_history = []
    for j in range(len(alpha)):
        avg_cost_history.append([])
        complete_history.append([])
        migration_overhead_history.append([])
    """
    avg_tf = np.zeros(len(alpha))
    migration_overhead = np.zeros(len(alpha))
    complete = np.zeros(len(alpha))
    while run > 0:
        users = create_users(number_of_user, cur_time=start_time)
        set_unit_migration_overhead(random.uniform(test[i][0], test[i][1]))
        for k in range(len(alpha)):
            this_user = copy.deepcopy(users)
            if change[k] >= 1:
                label = "alpha = " + str(alpha[k])
            else:
                label = "Random Validation"
            edge = create_edge(evn, number_of_edge=number_of_edge, time_interval=int(30),
                               channel=[11, 34, 17, 18, 19, 20, 11, 12, 34, 16],
                               number_of_core=2,
                               bandwidth_max=30)
            tf, rf, cp, _, _, _, _= \
                compare(run, alpha[k], None, label, evn, edge, this_user, number_of_edge=number_of_edge,
                        number_of_user=number_of_user, change=change[k], verbose=1,
                        time_interval=int(30), system_interval=30, start_time=start_time,
                        test_duration=3000, system_timer=1)
            migration_overhead[k] += rf
            avg_tf[k] += tf * 1000
            complete[k] += cp * 100
            print("run", run, "finish", alpha[k])
        run = run - 1
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("run", run)
        print("a4=", list(migration_overhead))
        print("a4=", list(avg_tf))
        print("a4=", list(complete))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
    for k in range(len(alpha)):
        migration_overhead[k] = migration_overhead[k] / number_of_run
        avg_tf[k] = avg_tf[k] / number_of_run
        complete[k] = complete[k] / number_of_run
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("a4=",list(migration_overhead))
    print("a4=",list(avg_tf))
    print("a4=",list(complete))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>")

    """"
    font_size = 14
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)     # fontsize of the x and y labels
    plt.rc('legend', fontsize=14)           # legend fontsize
    """

    plt.subplot(222)
    plt.plot(alpha, avg_tf, marker="+")

    plt.subplot(221)
    plt.plot(alpha, migration_overhead, marker="+")

    plt.subplot(223)
    plt.plot(alpha, complete, marker="+")



"""
plt.subplot(224)
plt.xlabel("alpha")
plt.ylabel("Job Success [%]")
plt.legend()
plt.grid(True)
"""

plt.subplot(222)
plt.xlabel("alpha")
plt.ylabel("TF/JOB [ms]")
plt.legend()
plt.grid(True)

plt.subplot(221)
plt.xlabel("alpha")
plt.ylabel("Migration Overhead [s]")
plt.legend()
plt.grid(True)

plt.subplot(223)
plt.xlabel("alpha")
plt.ylabel("Job Success [%]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

for n in range(number_of_user):
    users[n].simple_summary()
