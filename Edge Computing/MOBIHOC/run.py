import random
from worker import *
import numpy as np
import time
import math
import threading
import copy
from concurrent.futures import ThreadPoolExecutor


def initial_energy_all_local(selection, player):
    energy, finished = 0, 0
    user_hist = [[] for n in range(player.number_of_user)]
    # print(selection)
    ee = []
    for n in range(player.number_of_user):
        player.users[n].local_only_execution()
        if selection[n] == -1:
            energy += player.users[n].local_only_energy
            ee.append(round(player.users[n].local_only_energy, 5))
            if player.users[n].local_only_enabled:
                finished += 1
            user_hist[n].append(player.users[n].local_only_energy)
    # print(selection, "finished", finished, "energy", energy)
    return ee, finished, user_hist


def energy_update(player, selection, user_hist, save=True):
    energy, finished, transmission, computation, edge_computation = [], [], [], [], []
    for n in range(player.number_of_user):
        if selection[n] == -1:
            if save:
                user_hist[n].append(round(player.users[n].local_only_energy, 5))
            energy.append(round(player.users[n].local_only_energy, 5))
            if player.users[n].local_only_enabled:
                finished.append(1)
            else:
                finished.append(0)
            transmission.append(0)
            computation.append(0)
            edge_computation.append(0)
        else:
            config = player.users[n].config
            k = selection[n]
            if config is not None:
                e, f, tt, ct, et = player.users[n].remote_execution()
                energy.append(round(e, 5))
                transmission.append(round(tt, 4))
                computation.append(round(ct, 4))
                edge_computation.append(round(et, 4))
                finished.append(f)
                if save:
                    user_hist[n].append(round(e, 5))
            else:
                transmission.append(0)
                computation.append(0)
                edge_computation.append(0)
                finished.append(0)
    # if np.sum(finished) == player.number_of_user:
    return user_hist, energy, finished, transmission, computation, edge_computation


def get_request(channel_allocation, just_updated, player, selection, full, epsilon):
    for n in range(player.number_of_user):
        player.users[n].partition()

    D_n = np.array([player.users[n].DAG.D/1000 for n in range(player.number_of_user)])
    X_n = np.array([player.users[n].remote for n in range(player.number_of_user)])
    Y_n = np.array([player.users[n].local for n in range(player.number_of_user)])
    user_cpu = np.array([player.users[n].freq for n in range(player.number_of_user)])
    edge_cpu = np.array([player.edges[k].freq for k in range(player.number_of_edge)])
    number_of_chs = np.array([player.edges[k].number_of_chs for k in range(player.number_of_edge)])
    P_max = np.array([player.users[n].p_max for n in range(player.number_of_user)])
    B = np.array([player.users[n].local_to_remote_size for n in range(player.number_of_user)])
    H = np.array([[player.users[n].H[k] for k in range(player.number_of_edge)] for n in range(player.number_of_user)])

    info = {
        "selection": selection,
        "number_of_edge": player.number_of_edge,
        "number_of_user": player.number_of_user,
        "D_n": D_n,
        "X_n": X_n,
        "Y_n": Y_n,
        "user_cpu": user_cpu,
        "edge_cpu": edge_cpu,
        "number_of_chs": number_of_chs,
        "P_max": P_max,
        "B": B,
        "H": H,
        "W": 2 * math.pow(10, 6),
        "who": None,
        "full": full,
        "default_channel": 1,
        "channel_allocation": channel_allocation,
        "step": 0.01,
        "interval": 10,
        "stop_point": epsilon
    }
    reset_request_pool(player.number_of_user)
    start = time.time()

    copied_info = []
    for n in range(player.number_of_user):
        info["who"] = player.users[n]
        copied_info.append(copy.deepcopy(info))

    with ThreadPoolExecutor(max_workers=player.number_of_user) as executor:
        executor.map(worker, copied_info)

    """
    for n in range(player.number_of_user):
        # 为每个worker创建一个线程
        info["who"] = player.users[n]
        # x = threading.Thread(target=worker, args=(copy.deepcopy(info),))
        # x.start()
        worker(copy.deepcopy(info))
    """
    while not check_worker(player.number_of_user):
        t = 0

    opt_delta = []
    for n in range(player.number_of_user):
        if player.users[n].config is not None:
            opt_delta.append(player.users[n].config[5])
        else:
            opt_delta.append(-1)

    print("request finished in >>>>>>>>>>>>>>>>", time.time() - start, selection, opt_delta)
    not_tested = [n for n in range(player.number_of_user)]
    n = 0
    while len(not_tested) > 0:
        # n = random.choice(not_tested)
        #if n == just_updated:
            #continue
        if get_request_pool()[n] is not None:
            return get_requests(get_request_pool()[n], selection)
        else:
            not_tested.remove(n)
            n += 1
    return None
