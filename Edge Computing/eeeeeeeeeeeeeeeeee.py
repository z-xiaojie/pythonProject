from itertools import product
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from mlxtend.plotting import ecdf
from statsmodels.distributions.empirical_distribution import ECDF


def show(a, b, c, d):
    plt.subplot(221)
    plt.plot(a, marker="X")

    plt.subplot(222)
    plt.plot(b, marker="X")

    plt.subplot(223)
    plt.plot(c, marker="X")

    plt.subplot(224)
    plt.plot(d, marker="X")


def cal(a , b, c, number_of_run):
    for k in range(len(alpha)):
        a[k] = a[k] / number_of_run
        b[k] = b[k] / number_of_run
        c[k] = c[k] / number_of_run
    return a, b, c


def sun(a1, b1, c1, d1, num):
    a = np.zeros(len(alpha))
    b = np.zeros(len(alpha))
    c = np.zeros(len(alpha))
    d = np.zeros(len(alpha))
    for i in range(len(num)):
        for k in range(len(alpha)):
            a[k] += a1[i][k] / num[i]
            b[k] += b1[i][k] / num[i]
            c[k] += c1[i][k] / num[i]
            d[k] += d1[i][k] / num[i]
    a = a / len(num)
    b = b / len(num)
    c = c / len(num)
    d = d / len(num)
    return a, b, c, d

# local
"""
a= [991.3585203553764, 1547.0917416673853, 3350.0266901811096, 3276.838772881571, 4974.733264689678, 7070.206446604667, 39333.917005057156, 39587.445766280005]
b= [35753.58299926738, 35156.5479581243, 34765.395590821194, 34571.07790319285, 34705.57286808059, 34766.66621633939, 39702.61653431383, 47399.772832319715]
c= [2595.8044251994334, 2594.1498602818815, 2591.610500669131, 2594.9473163395187, 2594.3487598790443, 2591.888459982856, 2595.6633976161133, 2596.1301860300605]
d= [397988.0, 397645.0, 397538.0, 398215.0, 398349.0, 398295.0, 398511.0, 377264.0]
"""
a= [0.0, 0.0, 17.780288659611443, 29.706187436237027, 45.50606360009934, 1745.2425119698078, 4523.789967282531, 4544.376844239366]
b= [1443.3102882091812, 1371.674386859714, 1367.4511274366164, 1405.227859167966, 1337.215360738774, 1364.307002282997, 1678.1590864979462, 1704.1470446746152]
c= [30539.0, 30538.0, 30538.0, 30537.0, 30538.0, 30539.0, 30537.0, 30534.0]
d= [30560.0, 30560.0, 30560.0, 30560.0, 30560.0, 30560.0, 30560.0, 30560.0]


alpha = [0.3, 0.5, 0.7, 0.8, 0.9, 1 ,  1, 1]

a1, b1, c1, d1 = sun([a], [b], [c], [d], [2])

label = ["0.3", "0.5", "0.7", "0.8", "0.9",  "1", "V", "F"]

print(list(a1))
print(list(b1))
print(list(c1))
print(list(d1))

show(a1, b1, c1, d1)


plt.subplot(221)
plt.xlabel("alpha")
plt.ylabel("Migration Overhead [s]")
plt.xticks(np.arange(len(alpha)), label)
#plt.legend()
plt.grid(True)

plt.subplot(222)
plt.xlabel("alpha")
plt.ylabel("TF/JOB [ms]")
plt.xticks(np.arange(len(alpha)), label)
#plt.legend()
plt.grid(True)

plt.subplot(223)
plt.xlabel("alpha")
plt.ylabel("Job Success Percentage[%]")
plt.xticks(np.arange(len(alpha)), label)
#plt.legend()
plt.grid(True)

plt.subplot(224)
plt.xlabel("alpha")
plt.ylabel("Number of Success Job")
plt.xticks(np.arange(len(alpha)), label)
#plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()