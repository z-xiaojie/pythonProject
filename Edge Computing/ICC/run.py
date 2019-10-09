import random
from Role import Role
import numpy as np
import matplotlib.pyplot as plt
import copy
import math


def initial_energy_all_local(selection, player):
    energy, finished = 0, 0
    # print(selection)
    ee = []
    for n in range(player.number_of_user):
        player.users[n].local_only_execution()
        if selection[n] == -1:
            energy += player.users[n].local_only_energy
            ee.append(round(player.users[n].local_only_energy, 5))
            if player.users[n].local_only_enabled:
                finished += 1
    # print(selection, "finished", finished, "energy", energy)
    return ee


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
                e, f, tt, ct, et = player.users[n].remote_execution(player.edges[k].freq,
                                                                    player.edges[k].get_bandwidth(n), k)
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
    return user_hist, energy, finished, transmission, computation, edge_computation

