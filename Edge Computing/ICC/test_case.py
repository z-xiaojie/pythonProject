import random
from Role import Role
from run import initial_energy_all_local, energy_update
import numpy as np
import matplotlib.pyplot as plt
import copy
import math


def test(x, full, model=0, epsilon=0.001, number_of_user=5, number_of_edge=1, player=None, network=None, cpu=None, d_cpu=None, H=None, ):

    network = network
    cpu = cpu

    selection = np.zeros(number_of_user).astype(int) - 1
    ee_local = initial_energy_all_local(selection, player)

    t = 0
    hist = []
    user_hist = [[] for n in range(number_of_user)]
    pre_energy = np.sum(ee_local)
    ttt = 0
    while t < 1000:
        changed = False
        changed_k = -1
        for n in range(number_of_user):
            validation = []
            for k in range(number_of_edge):
                config = player.users[n].select_partition(full, epsilon, k, f_e=player.edges[k].freq,
                                                          bw=player.edges[k].get_bandwidth(n))
                if config is not None:
                    validation.append({
                        "edge": k,
                        "config": config
                    })
            # print("user", n, "validation", validation)
            if len(validation) > 0:
                #if selection[n] == validation[0]["edge"]:
                    #continue
                validation.sort(key=lambda x: x["config"][1])
                if selection[n] != validation[0]["edge"]:
                    # selfish update, the overall energy consumption is not considered.
                    if model == 1:
                        player.users[n].config = validation[0]["config"]
                        if selection[n] != -1:
                            player.edges[selection[n]].tasks.remove(player.users[n])
                        player.edges[validation[0]["edge"]].tasks.append(player.users[n])
                        selection[n] = validation[0]["edge"]
                        changed = True
                        break
                    # add n to its optimal edge node
                    player.edges[validation[0]["edge"]].tasks.append(player.users[n])
                    if selection[n] != -1:
                        player.edges[selection[n]].tasks.remove(player.users[n])
                    # validation test, if all pre-existing users can still finish their task
                    configs = [player.users[i].config for i in range(number_of_user)]
                    new_configs = []
                    allowed = True
                    k1, k2 = validation[0]["edge"], selection[n]
                    for j in range(number_of_user):
                        if (selection[j] == k1 or (selection[j] == k2 and k2 != -1)) and j != n:
                            # pre_config = player.users[j].config
                            temp_config = player.users[j].select_partition(full, epsilon, validation[0]["edge"]
                                                                           ,f_e=player.edges[validation[0]["edge"]].freq,
                                                                           bw=player.edges[validation[0]["edge"]].get_bandwidth(j))
                            new_configs.append(temp_config)
                            # negative impact to the overall system, update disabled
                            if temp_config is None:
                                allowed = False
                            #else:
                            #    player.users[j].config = temp_config
                    if not allowed:
                        if selection[n] != -1:
                            player.edges[selection[n]].tasks.append(player.users[n])
                        player.edges[validation[0]["edge"]].tasks.remove(player.users[n])
                        # print(">>>> ||||||||| stop ||||||| user", n, validation[0])
                        continue
                    else:
                        k1, k2 = validation[0]["edge"], selection[n]
                        for j in range(number_of_user):
                            if (selection[j] == k1 or (selection[j] == k2 and k2 != -1)) and j != n:
                                # player.users[j].config = new_configs[j]
                                player.users[j].config = player.users[j].select_partition(full, epsilon, validation[0]["edge"]
                                                                  ,f_e=player.edges[validation[0]["edge"]].freq,
                                                                  bw=player.edges[validation[0]["edge"]].get_bandwidth(j))
                    if model == 0:
                        old_edge_id, old_config = selection[n], player.users[n].config
                        selection[n], player.users[n].config = validation[0]["edge"], validation[0]["config"]
                        _, new_energy, _, _, _, _ = energy_update(player, selection, user_hist, save=False)
                        if np.sum(new_energy) > pre_energy:
                            selection[n], player.users[n].config = old_edge_id, old_config
                            player.edges[validation[0]["edge"]].tasks.remove(player.users[n])
                            # print("||||||||| stop ||||||| user", n, "new energy", np.sum(new_energy), "old", pre_energy)
                            k1, k2 = validation[0]["edge"], selection[n]
                            for j in range(number_of_user):
                                if (selection[j] == k1 or (selection[j] == k2 and k2 != -1)) and j != n:
                                    player.users[j].config = configs[j]
                            if selection[n] != -1:
                                player.edges[selection[n]].tasks.append(player.users[n])
                            continue
                        else:
                            player.users[n].config = validation[0]["config"]
                            # print("usre", n, "local E", player.users[n].local_only_energy, "select edge node",
                            #     validation[0]["edge"], "with", validation[0]["config"])
                            # if selection[n] != -1:
                            #    player.edges[selection[n]].tasks.remove(player.users[n])
                            selection[n] = validation[0]["edge"]
                            changed = True
                            # changed_k = validation[0]["edge"]
                            # selfish update, the overall energy consumption is not considered.
                            break
            else:
                if selection[n] != -1:
                    changed = True
                    changed_k = selection[n]
                    print("user", n, "run task locally", player.users[n].local_only_energy,
                          player.users[n].local_only_enabled)
                    player.edges[selection[n]].tasks.remove(player.users[n])
                    selection[n] = -1
                    break
                selection[n] = -1
                # print("user", n, "run task locally", player.users[n].local_only_energy,
                #     player.users[n].local_only_enabled)

        for n in range(number_of_user):
            if selection[n] != -1:
                f_e = player.edges[selection[n]].freq
                bw = player.edges[selection[n]].get_bandwidth(n)
                player.users[n].config = player.users[n].select_partition(full, epsilon, selection[n], f_e=f_e, bw=bw)

        user_hist, energy, finished, transmission, computation, edge_computation = energy_update(player, selection, user_hist)

        pre_energy = np.sum(energy)
        # print(selection, "finished", finished, "energy", np.sum(pre_energy))
        print(t, "energy", energy)
        # print("bandwidth", [ math.pow(10, -6) * player.edges[k].bandwidth/len(player.edges[k].tasks) for k in range(player.number_of_edge)])
        t += 1
        hist.append(np.sum(energy))
        if not changed:
            ttt += 1
        if ttt >= 10:
            break

   # print("cpu", cpu / math.pow(10, 9))
   # print("network", network / math.pow(10, 6))

    for n in range(number_of_user):
        if selection[n] != -1:
            f_e = player.edges[selection[n]].freq
            bw = player.edges[selection[n]].get_bandwidth(n)
            player.users[n].config = player.users[n].select_partition(full, epsilon, selection[n], f_e=f_e, bw=bw)

    user_hist, energy, finished, transmission, computation, edge_computation = energy_update(player, selection,
                                                                                             user_hist)

    """
    for n in range(number_of_user):
        print([round(player.users[n].DAG.jobs[i].output_data / 8000, 0) for i in range(len(player.users[n].DAG.jobs))]
              , [round(player.users[n].DAG.jobs[i].computation / math.pow(10, 9), 4) for i in
                 range(len(player.users[n].DAG.jobs))])
    """

    # print(selection)
    # print(finished, "finished", np.sum(finished))
    # print("max      local CPU", [round(player.users[n].freq / math.pow(10, 9), 4) for n in range(number_of_user)])

    opt_cpu = []
    opt_power = []
    opt_delta = []
    bandwidth = []
    for n in range(number_of_user):
        if player.users[n].config is not None:
            opt_cpu.append(round(player.users[n].config[2] / math.pow(10, 9), 4))
            opt_power.append(round(player.users[n].config[3] * math.ceil(player.edges[selection[n]].get_bandwidth(n)/math.pow(10, 6)), 4))
            opt_delta.append(player.users[n].config[0])
            bandwidth.append(math.ceil(player.edges[selection[n]].get_bandwidth(n)/math.pow(10, 6)))
        else:
            opt_delta.append(-1)
            opt_power.append(0)
            opt_cpu.append(0)
            bandwidth.append(0)

    print(">>>>>>>>>>>>>>>> TIME >>>>>>>>>>>>>>>>>>")
    print("adjusted local power", opt_power)
    print("adjusted local   CPU", opt_cpu)
    # print("            opt_delta", opt_delta)
    print("data", [round(player.users[n].local_to_remote_size / 8000, 5) for n in range(player.number_of_user)])

    #print("              deadline", [round(player.users[n].DAG.D / 1000, 4) for n in range(number_of_user)])
    #print("           finish time", list(np.round(np.array(transmission) + np.array(computation) + np.array(edge_computation),4)))
    print("     transmission time", transmission)
    print("     computation  time", computation)
    print("edge computation  time", edge_computation)
    """
    print(">>>>>>>>>>>>>>>> energy >>>>>>>>>>>>>>>>>>")
    print("edge based energy", round(np.sum(energy), 6), energy)
    print("local only energy", round(np.sum(ee_local), 6), ee_local)
    """
    print("local computation",
          [round(player.users[n].local/math.pow(10, 9), 5) for n in range(player.number_of_user)])
    print("remote computation",
          [round(player.users[n].remote/math.pow(10, 9), 5) for n in range(player.number_of_user)])

    # print("improvement", 1 - round(np.sum(energy), 5) / round(np.sum(ee_local), 5))
    #p rint(finished, "finished", np.sum(finished))

    plt.subplot(1, 2, x)
    plt.plot(hist, label="overall")

    for n in range(number_of_user):
        plt.plot(user_hist[n], label="user"+str(n+1))
        # print("u"+str(n)+"=", user_hist[n])
    # print("all=", list(hist))
    print("total computation", [round(player.users[n].total_computation, 4) for n in range(number_of_user)])
    print("finish time", [round((transmission[n] + computation[n] + edge_computation[n] - player.users[n].DAG.D/1000), 5) for n in range(player.number_of_user)])

    matched = 0
    for n in range(number_of_user):
        validation = []
        for k in range(number_of_edge):
            config = player.users[n].select_partition(full, epsilon, k, f_e=player.edges[k].freq,
                                                      bw=player.edges[k].get_bandwidth(n))
            if config is not None:
                validation.append({
                    "edge": k,
                    "config": config
                })
        # print("user", n, "validation", validation)
        if len(validation) > 0:
            validation.sort(key=lambda x: x["config"][1])
            if selection[n] == validation[0]["edge"]:
                matched += 1
        else:
            if selection[n] == -1:
                matched += 1

    print("matched", matched)
    # for n in range(number_of_user):
       # print(player.users[n].config)

    #plt.legend()

    return bandwidth, opt_delta, selection, np.sum(finished)/number_of_user, round(np.sum(energy), 5), round(np.sum(ee_local), 5), 1 - round(np.sum(energy), 5) / round(np.sum(ee_local), 5)
