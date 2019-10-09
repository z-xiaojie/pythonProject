import random
from Role import Role
from run import initial_energy_all_local
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from test_case import test

iterations = 10
I = 10
hist = [ [np.zeros(I) for i in range(20)] for j in range(3)]
selection1, selection2, selection3 = None, None, None
opt_delta1, opt_delta2 = None, None
bandwidth1, bandwidth2 = None, None
for i in range(iterations):
    number_of_user, number_of_edge, epsilon = 10, 3, 0.01
    for t in range(I):
        network = np.array([random.uniform(10, 20) * math.pow(10, 6) for x in range(number_of_edge)])
        cpu = np.array([random.uniform(2.5, 3.5) * math.pow(10, 9) for x in range(number_of_edge)])
        H = [[np.random.rayleigh(np.sqrt(2 / np.pi) * math.pow(10, -3)) for y in range(number_of_edge)] for x in
             range(number_of_user)]
        d_cpu = np.array([random.uniform(1.8, 2.5) * math.pow(10, 9) for x in range(number_of_user)])
        player = Role(number_of_edge=number_of_edge, number_of_user=number_of_user, network=network, cpu=cpu,
                      d_cpu=d_cpu,
                      H=H)
        player.initial_DAG()
        bandwidth1, opt_delta1, selection1, finished, energy, local, improvement = test(1, False, model=1, epsilon=epsilon, number_of_user=number_of_user, number_of_edge=number_of_edge
                                     , player=copy.deepcopy(player), network=network, cpu=cpu)
        hist[0][0][t] += finished
        hist[0][1][t] += improvement
        hist[0][2][t] += energy
        hist[0][3][t] += local



        bandwidth2, opt_delta2, selection2, finished, energy, local, improvement = test(2, True, model=1, epsilon=epsilon, number_of_user=number_of_user,
                                     number_of_edge=number_of_edge
                                     , player=copy.deepcopy(player), network=network, cpu=cpu)
        hist[1][0][t] += finished
        hist[1][1][t] += improvement
        hist[1][2][t] += energy
        hist[1][3][t] += local

        """
        selection3, finished, energy, local, improvement = test(model=2, epsilon=epsilon, number_of_user=number_of_user,
                                                                number_of_edge=number_of_edge
                                                                , player=copy.deepcopy(player), network=network,
                                                                cpu=cpu)
        hist[2][0][t] += finished
        hist[2][1][t] += improvement
        hist[2][2][t] += energy
        hist[2][3][t] += local
        """
        number_of_user += 3

print(selection1, "finished", hist[0][0]/iterations)
print(selection2, "finished", hist[1][0]/iterations)
print(">>>>>>>>>>> partition>>>>>>>>>>")
print(opt_delta1, "finished", hist[1][0]/iterations)
print(opt_delta2, "finished", hist[1][0]/iterations)
print(">>>>>>>>>>> bandwidth1>>>>>>>>>>")
print(bandwidth1)
print(bandwidth2)
print("average improvement", hist[0][1]/iterations, hist[0][2]/iterations, hist[0][3]/iterations)
print("average improvement", hist[1][1]/iterations, hist[1][2]/iterations, hist[1][3]/iterations)


"""
print(selection1, "finished", hist[2][0]/iterations)
print("average improvement", hist[2][1]/iterations, hist[2][2]/iterations, hist[2][3]/iterations)
"""
# print("improvement", 1 - (hist[0][2]/iterations)/(hist[1][2]/iterations))#, 1 - (hist[0][2]/iterations)/hist[2][2]/iterations)


# plt.show()
