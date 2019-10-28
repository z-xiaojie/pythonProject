import random
from Role import Role
import numpy as np
import matplotlib.pyplot as plt
import copy
import math


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


def update_config(full, this_n, player, k, epsilon, selection):
    for n in range(player.number_of_user):
        if selection[n] == k and this_n != n:
            f_e = player.edges[selection[n]].get_freq(n)
            bw = player.edges[selection[n]].get_bandwidth(n)
            config = player.users[n].select_partition(full, epsilon, selection[n], f_e=f_e, bw=bw)
            if config is not None:
                player.users[n].config = config


def get_request(model, just_updated, player, selection, full, epsilon):
    request = []
    not_tested = [n for n in range(player.number_of_user)]
    while len(not_tested) > 0:
        n = random.choice(not_tested)
        if n == just_updated:
            continue
        validation = []
        for k in range(player.number_of_edge):
            config = player.users[n].select_partition(full, player.edges[k], epsilon=epsilon, p_adjust=0.5, default_channel=3)
            if config is not None:
                validation.append({
                    "edge": k,
                    "config": config
                })
        if len(validation) > 0:
            validation.sort(key=lambda x: x["config"][0])
            # print(n, "set config", validation[0])
            if validation[0]["edge"] != selection[n]:
                request.append({
                    "user": n,
                    "validation": validation[0],
                    "local": False
                })
                return request
            else:
                if model != 2:
                    if math.fabs(validation[0]["config"][0] - player.users[n].config[0]) >= player.users[n].config[0] * 0.01:
                        request.append({
                            "user": n,
                            "validation": validation[0],
                            "local": False
                        })
                        return request
                    # player.users[n].config = config
        else:
            if selection[n] != -1:
                request.append({
                    "user": n,
                    "validation": None,
                    "local": True
                })
                return request
        not_tested.remove(n)
    return request
