import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Compare import compare
from Main import create_edge
from User import User
import copy
from Optimization import set_unit_migration_overhead
from Prediction import assign_channel_to_edge


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
        [0.85, 4, 6, 6],
        [0.64, 5, 6, 6],
        [0.77, 4, 5, 5],
        [0.62, 3, 6, 6],
        [0.87, 5, 6, 6],
        [0.76, 3, 6, 6],
        [0.89, 3, 5, 5],
        [0.85, 3, 7, 7],
        [0.76, 5, 5, 5],
        [0.5, 4, 7, 7],
        [0.86, 4, 7, 7],
        [0.77, 3, 6, 6],
        [0.89, 5, 6, 6],
        [0.89, 3, 7, 7],
        [0.98, 5, 7, 7],
        [0.97, 3, 7, 7],
        [0.95, 4, 7, 7],
        [0.57, 3, 7, 7],
        [0.58, 4, 5, 5],
        [0.53, 3, 7, 7],
        [0.52, 4, 5, 5],
        [0.87, 4, 6, 6],
        [0.84, 3, 5, 5],
        [0.86, 4, 6, 6],
        [0.5, 4, 7, 7],
        [0.82, 4, 5, 5],
        [0.53, 5, 7, 7],
        [0.62, 5, 6, 6],
        [0.69, 5, 6, 6],
        [0.98, 4, 5, 5]
    ])
    for n in range(number_of_user):
        #users.append(User(cur_time, n, param[n][0], param[n][1], param[n][2], param[n][3]))
        T = np.random.randint(3, 8)
        users.append(User(cur_time, n, round(np.random.uniform(0.5, 1), 2), np.random.randint(3, 6),
                         T, T))
        #users[-1].simple_summary()
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

"""
a = 1, a=0.8, a=0.3, a=0, a=-1, a=-1
a40 = [0.9797212378404323, 0.9794203015708023, 0.9794203015708023, 0.9794203015708023, 0.9748574465227723]
a50 = [0.9756912757795227, 0.9741210028959907, 0.9741210028959907, 0.9741210028959907, 0.9686396054186416]
a60 = [0.9695114817843455, 0.9420329315751371, 0.9420329315751371, 0.9420329315751371, 0.941251565372568]
a70 = [0.9609425287001442, 0.9095287559131274, 0.9095287559131274, 0.9095287559131274, 0.7567687826971023]
a80 = [0.9562094296300888, 0.7860155580749512, 0.7860155580749512, 0.7860155580749512, 0.600001878401819]
a90 = [0.9353741094579235, 0.7013767967341215, 0.7013767967341215, 0.7013767967341215, 0.5925325879837448]
"""
########################
#  performance compare
#######################
evn = load_network_evn()
number_of_user = 70
number_of_edge = 10
"""
tf = 10
rf = 10
t = a * 10 + (1-a)*10
"""
# test case 1
"""
alpha  = [1,0.8,0.3,0,1,]
change = [1,1  ,1  ,1,-1]

# test case 1
alpha  = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1, 1, 1]
change = [1, 1,   1,   1,   1,   1,   1,   1,   1,   1,    1, 0, -1]

alpha  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
change = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
"""
alpha  = [0.5, 0.5]
change = [2, 6]

marker = ["+", "1", "2", "3", "4", "+", "x", "|", "*"]

avg_tf = np.zeros(len(alpha))
migration_overhead = np.zeros(len(alpha))
migration_overhead_fail = np.zeros(len(alpha))
complete = np.zeros(len(alpha))
jobs = np.zeros(len(alpha))
complete_history = np.zeros(len(alpha))

# time line
avg_cost_history = []
none_service_history = []
complete_history_time = []
for k in range(len(alpha)):
    avg_cost_history.append(np.zeros(10))
    none_service_history.append(np.zeros(10))
    complete_history_time.append(np.zeros(10))
avg_cost_history = np.array(avg_cost_history)
none_service_history = np.array(none_service_history)
complete_history_time = np.array(complete_history_time)

# allocate channel to each edge
# [8, 9,19, 21 ,22, 23,24,25, 11, 14, 7] for 60 Mhz
# [11, 34, 17, 18, 19, 20, 11, 12, 34, 16] for 40 Mhz
interval = 30
test = [0.75]
number_of_run = 3
run = number_of_run
start_time = int(80000)
while run > 0:
    channel = [11, 34, 17, 18, 19, 20, 11, 12, 34, 16]  #np.random.randint(0, 26, 11)
    avg_data = []
    for k in range(number_of_edge):
        avg_data.append(assign_channel_to_edge(evn, time_interval=int(30), ch=channel[k]))
    users = create_users(number_of_user, cur_time=start_time)
    set_unit_migration_overhead(test[0])
    for k in range(len(alpha)):
        this_user = copy.deepcopy(users)
        edge = create_edge(evn, number_of_edge=number_of_edge, avg_data=avg_data, time_interval=int(interval),
                           channel=channel,
                           number_of_core=2,
                           bandwidth_max=30)
        tf, rf, cp, tf_h, cp_h, _, nj, ns_h, cp_h_t, rf_f= \
            compare(run, alpha[k], None, None, evn, edge, this_user, number_of_edge=number_of_edge,
                    number_of_user=number_of_user, change=change[k], verbose=1,
                    time_interval=int(interval), system_interval=300, start_time=start_time,
                    test_duration=3000, system_timer=1)
        migration_overhead[k] += rf
        migration_overhead_fail[k] += rf_f
        avg_tf[k] += tf
        complete[k] += cp
        jobs[k] += nj
        avg_cost_history[k] = avg_cost_history[k] + tf_h
        complete_history[k] = complete_history[k] + cp_h
        complete_history_time[k] = complete_history_time[k] + cp_h_t
        # none_service_history[k] = none_service_history[k] + ns_h
        print("run", run, "finish", alpha[k])
    run = run - 1
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("run", run, "number_of_user", number_of_user)
    print("total_rf=", list(migration_overhead / (number_of_run - run)))
    print("total_rf_f=", list(migration_overhead_fail / (number_of_run - run)))
    print("avg_tf=", list(avg_tf / (number_of_run - run)))
    print("completed_job=", list(complete / (number_of_run - run)))
    print("transmitted_job=", list(jobs / (number_of_run - run)))
    for k in range(len(alpha)):
        print("avf_tf_history" + str(alpha[k]) + "=", list(avg_cost_history[k] / (number_of_run - run)))
    print("completed_job=", list(complete_history / (number_of_run - run)))
    print("completed_job_history=", list(complete_history_time / (number_of_run - run)))
    for k in range(len(alpha)):
         print("cp_history" + str(alpha[k]) + "=", list(complete_history_time[k] / (number_of_run - run)))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>")

for k in range(len(alpha)):
    migration_overhead[k] = migration_overhead[k] / number_of_run
    migration_overhead_fail[k] = migration_overhead_fail[k] / number_of_run
    avg_tf[k] = avg_tf[k] / number_of_run
    complete[k] = complete[k] / number_of_run
    jobs[k] = jobs[k] / number_of_run
    avg_cost_history[k] = avg_cost_history[k] / number_of_run
    complete_history[k] = complete_history[k] / number_of_run
    complete_history_time[k] = complete_history_time[k] / number_of_run
    # none_service_history[k] = none_service_history[k] / number_of_run
print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
print("a=", list(migration_overhead))
print("h=", list(migration_overhead_fail))
print("b=", list(avg_tf))
print("c=", list(complete))
print("d=", list(jobs))
print("e=", list(avg_cost_history))
print("g=", list(complete_history))
print("f=", list(complete_history_time))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>")



#for n in range(number_of_user):
    #users[n].simple_summary()
