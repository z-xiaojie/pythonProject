import random
from Role import Role
from run import initial_energy_all_local
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from test_case import test

iterations = 1
I = 1
hist = [ [np.zeros(I) for i in range(20)] for j in range(3)]
selection1, selection2, selection3 = None, None, None
opt_delta1, opt_delta2 = None, None
bandwidth1, bandwidth2 = None, None
for i in range(iterations):
    # 3, 40， 85
    # 3, 35,  101
    # 3, 30,  85, 87, 72
    number_of_user, number_of_edge, epsilon = 15, 3, 0.01
    chs = 15
    t = 0
    f = 1.25
    while t < I:
        number_of_chs = np.array([random.randint(15, 30) for x in range(number_of_edge)])
        cpu = np.array([random.uniform(4.5, 7.5) * math.pow(10, 9) for x in range(number_of_edge)])
        H = [[np.random.rayleigh(np.sqrt(2 / np.pi) * math.pow(10, -3)) for y in range(number_of_edge)] for x in
             range(number_of_user)]
        d_cpu = np.array([random.uniform(1.5, 2.5) * math.pow(10, 9) for x in range(number_of_user)])
        player = Role(number_of_edge=number_of_edge, number_of_user=number_of_user, epsilon=epsilon, number_of_chs=number_of_chs, cpu=cpu,
                      d_cpu=d_cpu,
                      H=H)
        player.initial_DAG()
        it, finish_hist1, bandwidth1, opt_delta1, selection1, finished, energy, local, improvement \
            = test(1, False, model=1, epsilon=epsilon, number_of_user=number_of_user, number_of_edge=number_of_edge
                                     ,player=copy.deepcopy(player))

        if finished != 1:
            print(">>>>>>>", np.sum(finished))
            # continue

        hist[0][0][t] += finished
        hist[0][1][t] += improvement
        hist[0][2][t] += energy
        hist[0][3][t] += local
        hist[0][4][t] += it

        """
        it, finish_hist2, bandwidth2, opt_delta2, selection2, finished, energy, local, improvement = test(2, True, model=1, epsilon=epsilon, number_of_user=number_of_user,
                                        number_of_edge=number_of_edge
                                        ,player=copy.deepcopy(player), network=network, cpu=cpu)
        hist[1][0][t] += finished
        hist[1][1][t] += improvement
        hist[1][2][t] += energy
        hist[1][3][t] += local

        
        selection3, finished, energy, local, improvement = test(model=2, epsilon=epsilon, number_of_user=number_of_user,
                                                                number_of_edge=number_of_edge
                                                                , player=copy.deepcopy(player), network=network,
                                                                cpu=cpu)
        hist[2][0][t] += finished
        hist[2][1][t] += improvement
        hist[2][2][t] += energy
        hist[2][3][t] += local
        """

        # plt.subplot(1, 2, 2)
        # plt.plot(np.array(finish_hist1)/number_of_user)

        #plt.subplot(2, 2, 4)
        #plt.plot(finish_hist2)
        # chs += 5
        f += 0.5
        # number_of_user += 3
        t += 1

print(selection1, "finished", hist[0][0]/iterations)
print(selection2, "finished", hist[1][0]/iterations)
print(">>>>>>>>>>> partition>>>>>>>>>>")
print(opt_delta1, "finished", hist[0][0]/iterations)
print(opt_delta2, "finished", hist[1][0]/iterations)
print(">>>>>>>>>>> bandwidth1>>>>>>>>>>")
print(bandwidth1)
print(bandwidth2)
print("adaptive=", list(hist[0][2]/iterations))
print("full=", list(hist[1][2]/iterations))
print("local=", list(hist[0][3]/iterations))
print("it=", list(hist[0][4]/iterations))
print("H=", H)
print("cpu=", cpu)
print("chs=", chs)
"""
print(selection1, "finished", hist[2][0]/iterations)
print("average improvement", hist[2][1]/iterations, hist[2][2]/iterations, hist[2][3]/iterations)
"""
# print("improvement", 1 - (hist[0][2]/iterations)/(hist[1][2]/iterations))#, 1 - (hist[0][2]/iterations)/hist[2][2]/iterations)

# plt.show()
