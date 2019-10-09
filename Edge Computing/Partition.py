import matplotlib.pyplot as plt
from Main import Main
import random
import numpy as np
from History import History
from itertools import combinations_with_replacement
from itertools import product


def update_history_partition_set(edge, users, number_of_user, number_of_edge, partition_history):
    L = 0
    S = 0
    update = list()
    while L < 15 and S < 250:
        S += 1
        user_partition = []
        for n in range(number_of_user):
            user_partition.append(np.random.randint(0, number_of_edge))
        stored = False
        for history in partition_history:
            if history.partition == user_partition:
                stored = True
                break
        if not stored:
            density, total_delay, feasible = feasible_test(edge, users, number_of_user, user_partition, number_of_edge)
            #print(S, L, user_partition)
            if feasible:
                L += 1
                new_history = History(user_partition, density, total_delay)
                update.append(new_history)
                if len(partition_history) < 500:
                    partition_history.append(new_history)
                else:
                    partition_history.sort(key=lambda x: x.avg_density, reverse=True)
                    if partition_history[-1].avg_density > new_history.avg_density:
                        partition_history[-1] = new_history
    print("------------------------------------------------------------------------",len(partition_history))
    return partition_history


def feasible_test(edge, users, number_of_user, user_partition, number_of_edge):
    total_density = 0
    total_delay = 0
    user_partition = np.array(user_partition)
    for k in range(number_of_edge):
        density = 0
        number_of_assigned_user = (user_partition == k).sum()
        max_fair_rate = edge[k].avg_data[int(edge[k].cur_time / edge[k].interval)] / number_of_assigned_user

        for n in range(number_of_user):
            if user_partition[n] == k:
                users[n].set_allocated_rate(max_fair_rate)
                density += users[n].get_density_with_rate()
                total_delay += users[n].job_size / max_fair_rate
        total_density += density

        assigned_user = []
        for n in range(number_of_user):
            if user_partition[n] == k:
                assigned_user.append(users[n])
        edge[k].users = []
        edge[k].users = assigned_user
        avg_rate = edge[k].avg_data[int(edge[k].cur_time / edge[k].interval)]
        if len(assigned_user) == 0:
            edge[k].user_to_core = []
            continue
        if not edge[k].job_scheduling(avg_rate):
            return 0, 0, False
    return total_density, total_delay, True
