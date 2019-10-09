import numpy as np
from itertools import combinations_with_replacement
import random
import copy
from itertools import product
from Task.knapsack_prob import opt
import math

time_for_one_task = None


def set_unit_migration_overhead(rf):
    global time_for_one_task
    time_for_one_task = rf


def migration_overhead(p1, p2, number_of_user):
    global time_for_one_task
    rf = 0
    if p1 is None or p2 is None:
        return 0
    for n in range(number_of_user):
        if p1[n] != p2[n]:
            rf += time_for_one_task
    return rf


def optimal_partition(alpha, edge, users, number_of_user, valid, pre_partition, policy=None):
    min_partition = None
    min_value = 99999
    if pre_partition is None:
        pre_tf = 0
    else:
        pre_tf = estimate_data_transfer(edge, users, number_of_user, pre_partition)
    for item in valid:
        user_partition = np.array(item)
        rf = migration_overhead(user_partition, pre_partition, number_of_user)
        tf = estimate_data_transfer(edge, users, number_of_user, user_partition)
        total = alpha*(pre_tf - tf) - (1-alpha) * rf
        if total < min_value:
            min_value = total
            min_partition = user_partition
    return min_partition
    # 15 25 22


def feasible(edge, users, number_of_user, number_of_edge, user_partition):
    can_scheduling = True
    for k in range(number_of_edge):
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
            can_scheduling = False
    return can_scheduling


def estimate_data_transfer(edge, users, number_of_user, user_partition):
    tf = 0
    for n in range(number_of_user):
        k = user_partition[n]
        assigned_number = np.count_nonzero(np.array(user_partition) == k)
        time = int(edge[k].cur_time / edge[k].interval)
        avg_rate = edge[k].avg_data[time] * edge[k].bandwidth_max / assigned_number
        if avg_rate == 0:
            tf = 99999
        else:
            tf += users[n].job_size / avg_rate
    return tf


def my(users, edge, number_of_user, number_of_edge):
    c_edge = copy.copy(edge)
    unassigned_users = copy.copy(users)
    new_partition = []
    sum_bandwidth = 0
    for k in range(len(c_edge)):
        adjust = c_edge[k].avg_data[int(c_edge[k].cur_time / c_edge[k].interval)]
        c_edge[k].bandwidth_real = c_edge[k].bandwidth_max*adjust
        sum_bandwidth += c_edge[k].bandwidth_real
    c_edge.sort(key=lambda x: x.bandwidth_real, reverse=True)
    unassigned_users.sort(key=lambda x: x.job_size, reverse=True)
    sum_job_size = 0
    for n in range(number_of_user):
        sum_job_size += users[n].job_size
        new_partition.append(-1)
    assign_size = np.zeros(number_of_edge)
    assign_number = np.zeros(number_of_edge)
    for k in range(number_of_edge):
        assign_size[k] = sum_job_size * c_edge[k].bandwidth_real / sum_bandwidth
        assign_number[k] = math.ceil(number_of_user * c_edge[k].bandwidth_real / sum_bandwidth)
    #print(assign_number)
    #print(assign_size)
    just_assigned = []
    assigned_id = []
    job_size = np.zeros(number_of_edge)
    for k in range(number_of_edge):
        just_assigned.append([])
        assigned_id.append([])
    run = 1
    while len(unassigned_users) > 0 and run <= 2:
        for k in range(len(c_edge)):
            adjust = c_edge[k].avg_data[int(c_edge[k].cur_time / c_edge[k].interval)]
            for n in range(len(unassigned_users)):
                just_assigned[k].append(unassigned_users[n])
                job_size[k] += unassigned_users[n].job_size
                c_edge[k].users = just_assigned[k]
                user_to_core = c_edge[k].user_to_core
                # len(just_assigned[k]) < assign_number[k] and
                if (job_size[k] <= assign_size[k] or run >= 2) and c_edge[k].job_scheduling(adjust):
                    new_partition[unassigned_users[n].id] = find_k_by_id(edge, c_edge[k].id)
                    assigned_id[k].append(str(unassigned_users[n].id) + "(" + str(unassigned_users[n].job_size) + ")")
                    if run >= 2:
                        break
                else:
                    job_size[k] -= unassigned_users[n].job_size
                    just_assigned[k].remove(unassigned_users[n])
                    c_edge[k].user_to_core = user_to_core
                    c_edge[k].users = just_assigned[k]
            for item in just_assigned[k]:
                if item in unassigned_users:
                    unassigned_users.remove(item)
       # print(len(unassigned_users))
            #density, _ = c_edge[k].summary()
            #if run == 2:
                #print(k, len(unassigned_users), density, ">>>>>>>>>>", assigned_id[k], c_edge[k].bandwidth_real)
        run += 1
        #assign_number += 1
    return np.array(new_partition)


def find_k_by_id(edge, id):
    for k in range(len(edge)):
        if id == edge[k].id:
            return k
    return -1


def display_edge(user_partition, number_of_edge, edge, users):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for k in range(number_of_edge):
        assigned = []
        for i in range(len(user_partition)):
            if user_partition[i] == k:
                assigned.append(str(i)+"("+str(users[i].job_size)+")")
        print("edge", k, assigned, edge[k].avg_data[int(edge[k].cur_time / edge[k].interval)] * edge[k].bandwidth_max)


def brute(users, edge, number_of_user, number_of_edge):
    times = 0
    min_tf = 9999
    # 495 * 70 * 6
    number_of_job = 0
    for u in users:
        number_of_job += edge[0].interval / u.interval
    for roll in product(np.arange(number_of_edge), repeat=number_of_user):
        new_user_partition = list(roll)
        if feasible(edge, users, number_of_user, number_of_edge, new_user_partition):
            tf = number_of_job * estimate_data_transfer(edge, users, number_of_user,
                                                        new_user_partition) / number_of_user
            if tf < min_tf:
                min_tf = tf
            print(times, min_tf, new_user_partition)
        times += 1


# input from my function
def valid(alpha, policy, pre_tf, pre_feasible, edge, users, number_of_edge, number_of_user, new_user_partition, user_partition):
    rf, tf, max_fitness = get_fitness(alpha, edge, users, pre_tf, new_user_partition, user_partition, number_of_user)
    if max_fitness > 0 or user_partition is None or policy == 3:
        return new_user_partition, max_fitness, tf, rf
    """
    Minimal Switch to reduce migration overhead
    """
    if pre_tf > tf:
        if pre_feasible:
            max_fitness = 0
        else:
            max_fitness = -999
        best_partition = None
        for n in range(len(new_user_partition)):
            new_p = new_user_partition[n]
            old_p = user_partition[n]
            if new_p == old_p:
                continue
            new_user_partition[n] = old_p
            new_rf, new_tf, new_fitness = get_fitness(alpha, edge, users, pre_tf, new_user_partition, user_partition,
                                                      number_of_user)
            if not feasible(edge, users, number_of_user, number_of_edge, new_user_partition):
                new_user_partition[n] = new_p
                continue
            if new_fitness > max_fitness:
                max_fitness = new_fitness
                best_partition = copy.deepcopy(new_user_partition)
            #print(pre_feasible, new_fitness, round(new_rf, 3), round(pre_tf, 3), round(tf, 3), round(new_tf, 3))
        if best_partition is not None:
            rf, tf, fitness = get_fitness(alpha, edge, users, pre_tf, best_partition, user_partition,
                                          number_of_user)
            return new_user_partition, fitness, tf, rf
    return None, 0, 0, 0


def get_fitness(alpha, edge, users, pre_tf, new_user_partition, user_partition, number_of_user):
    rf = migration_overhead(new_user_partition, user_partition, number_of_user)
    tf = estimate_data_transfer(edge, users, number_of_user, new_user_partition)
    # print(tf)
    number_of_job = 0
    for u in users:
        number_of_job += edge[0].interval / u.interval
    fitness = number_of_job * alpha * (pre_tf - tf) / number_of_user - (
            1 - alpha) * rf
    return rf, tf, fitness


def global_optimization(alpha, users, edge, number_of_user, number_of_edge, user_partition, memory, bandwidth, policy=None):
    # save pre_config
    user_to_cores = []
    for k in range(number_of_edge):
        user_to_cores.append(edge[k].user_to_core)

    past = 0
    rf = 0
    rf_f = 0
    """
    u1 = [5, 6, 4, 3, 6, 3, 3, 8, 6, 4, 4, 8, 1, 9, 1, 6, 2, 6, 2, 6, 2, 0, 6, 0, 0, 7, 1, 5, 5, 7]
    u2 = [5, 2, 0, 7, 3, 9, 4, 4, 0, 4, 4, 1, 2, 6, 1, 3, 6, 6, 8, 2, 1, 0, 6, 6, 7, 3, 2, 5, 5, 7]

    display_edge(u1, number_of_edge, edge, users)

    display_edge(u2, number_of_edge, edge, users)
    """
    new_user_partition = None
    if policy == -1:
        new_user_partition = np.array([random.randint(0, number_of_edge - 1) for x in range(0, number_of_user)])
        rf = migration_overhead(user_partition, new_user_partition, number_of_user)
    elif policy == 0:
        times = 0
        while times < 200:
            new_user_partition = np.array([random.randint(0, number_of_edge - 1) for x in range(0, number_of_user)])
            if feasible(edge, users, number_of_user, number_of_edge, new_user_partition):
                break
            times += 1
        rf = migration_overhead(user_partition, new_user_partition, number_of_user)
    else:
        if user_partition is None:
            pre_tf = 0
            pre_feasible = False
        else:
            pre_tf = estimate_data_transfer(edge, users, number_of_user, user_partition)
            pre_feasible = feasible(edge, users, number_of_user, number_of_edge, user_partition)

        bandwidth.sort(reverse=True)

        """
        past_assignment, fitness, past_bandwidth = memory.search_past_assignment(alpha, pre_tf, edge, users,
                                                                                 number_of_edge, number_of_user
                                                                                 , user_partition, bandwidth,
                                                                             pre_feasible)
        """
        fitness = 0
        past_assignment = None
        if past_assignment is None:
            if policy == 1 or policy == 4:
                #ljbj_assignment = my(users, edge, number_of_user, number_of_edge)
                past, new_user_partition, fitness, tf, rf = opt(alpha, policy, None, pre_tf, pre_feasible, edge, users,
                                                                number_of_edge, number_of_user,
                                                                user_partition,
                                                                memory, bandwidth)
                #print(pre_feasible, round(tf,3))
            if policy == 2 or policy == 5 or policy == 6:
                new_user_partition = my(users, edge, number_of_user, number_of_edge)
                #new_user_partition, fitness, tf, rf = valid(alpha, policy, pre_tf, pre_feasible, edge, users, number_of_edge,
                                                            #number_of_user, new_user_partition, user_partition)
                #if new_user_partition is not None:
                    #memory.replace(new_user_partition, bandwidth, tf)
                rf, tf, fitness = get_fitness(alpha, edge, users, pre_tf, new_user_partition, user_partition,
                                                  number_of_user)

                if policy == 6 and fitness <= 0 and user_partition is not None:
                    tf_diff = []
                    for n in range(number_of_user):
                        if user_partition[n] == new_user_partition[n]:
                            continue
                        k = user_partition[n]
                        assigned_number = np.count_nonzero(np.array(user_partition) == k)
                        time = int(edge[k].cur_time / edge[k].interval)
                        p_avg_rate = edge[k].avg_data[time] * edge[k].bandwidth_max / assigned_number

                        k = new_user_partition[n]
                        assigned_number = np.count_nonzero(np.array(new_user_partition) == k)
                        time = int(edge[k].cur_time / edge[k].interval)
                        n_avg_rate = edge[k].avg_data[time] * edge[k].bandwidth_max / assigned_number

                        tf_diff.append([n, round(users[n].job_size / p_avg_rate - users[n].job_size / n_avg_rate,4)])

                    tf_diff.sort(key=lambda x: x[1])
                    for item in tf_diff:
                        k1 = user_partition[item[0]]
                        k2 = new_user_partition[item[0]]
                        new_user_partition[item[0]] = k1
                        _, _, fitness_new = get_fitness(alpha, edge, users, pre_tf, new_user_partition, user_partition,
                                                  number_of_user)
                        if fitness_new > 0 and feasible(edge, users, number_of_user, number_of_edge, new_user_partition):
                            print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
                            break
                        else:
                            new_user_partition[item[0]] = k2
                    #print(new_user_partition)
                #print(pre_feasible, round(tf, 3))
                #display_edge(new_user_partition, number_of_edge, edge, users)
            if fitness <= 0:
                if user_partition is not None and pre_feasible:
                    restore(edge, user_partition, users, user_to_cores, number_of_user, number_of_edge)
                    return past, 0, user_partition, 0
                else:
                    rf_f = rf
        else:
            if fitness <= 0:
                if user_partition is not None and pre_feasible:
                    restore(edge, user_partition, users, user_to_cores, number_of_user, number_of_edge)
                    return past, 0, user_partition, 0
            new_user_partition = past_assignment
            past = 1
    # restore pre_config
    if new_user_partition is None:
        restore(edge, user_partition, users, user_to_cores, number_of_user, number_of_edge)
    else:
        if not feasible(edge, users, number_of_user, number_of_edge, new_user_partition) and user_partition is not None\
                and (policy != -1 or policy != 0):
            restore(edge, user_partition, users, user_to_cores, number_of_user, number_of_edge)
        else:
            user_partition = new_user_partition
    return past, rf, user_partition, rf_f


def restore(edge, user_partition, users, user_to_cores, number_of_user, number_of_edge):
    for k in range(number_of_edge):
        edge[k].user_to_core = user_to_cores[k]
    for k in range(number_of_edge):
        assigned_user = []
        for n in range(number_of_user):
            if user_partition[n] == k:
                assigned_user.append(users[n])
        edge[k].users = []
        edge[k].users = assigned_user


def local_optimization(edge, number_of_edge):
    # rescheduling based on predicted rate adjust
    for k in range(number_of_edge):
        edge[k].job_scheduling(edge[k].avg_data[int(edge[k].cur_time / edge[k].interval)])


# alpha : importance of data transfer versus migration overhead
def partition(alpha, edge, users, number_of_user, number_of_edge, policy=False, exist_user_partition=None):
    #global partition_history
    if policy != "RE_CFG":
        possible = combinations_with_replacement(np.arange(0, number_of_edge), number_of_user)
        #possible = update_history_partition_set(edge, users, number_of_user, number_of_edge, partition_history)
        #print(len(list(possible)))
        valid = list()
        for item in possible:
            # job scheduling
            user_partition = np.array(item)
            """
            # update information
            density, delay, _ = feasible_test(edge, users, number_of_user, user_partition, number_of_edge)
            item.avg_density = (density + item.avg_density * item.freq) / (item.freq + 1)
            item.avg_delay = (delay + item.avg_delay * item.freq) / (item.freq + 1)
            """
            feasible = True
            for k in range(number_of_edge):
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
                    feasible = False
            if feasible:
                valid.append(user_partition)
                if policy == "RANDOM_VALID":
                    return user_partition
        if len(valid) > 0:
            min_partition = optimal_partition(alpha, edge, users, number_of_user, valid, exist_user_partition,
                                              policy=None)
            if min_partition is not None:
                return partition(alpha, edge, users, number_of_user, number_of_edge, policy="RE_CFG",
                                 exist_user_partition=min_partition)
        """
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("user partition:", user_partition)
        for k in range(number_of_edge):
            print("edge", k, edge[k].user_to_core)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        """
        return None
    else:
        for k in range(number_of_edge):
            assigned_user = []
            for n in range(number_of_user):
                if exist_user_partition[n] == k:
                    assigned_user.append(users[n])
            edge[k].users = []
            edge[k].users = assigned_user
            if len(assigned_user) > 0:
                edge[k].job_scheduling(edge[k].avg_data[int(edge[k].cur_time / edge[k].interval)])
        """
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("user partition:", exist_user_partition)
        for k in range(number_of_edge):
            print("edge", k, edge[k].user_to_core)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        """
        return exist_user_partition

