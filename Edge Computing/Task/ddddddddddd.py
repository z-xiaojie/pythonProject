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
"""
# local
a= [979.7849011868174, 1595.906647753989, 3711.6064404106046, 3244.1063660039727, 5439.158471583483, 6091.889055199302, 45703.09911840257, 46102.30256170288]
b= [43451.84915646989, 42473.73447175789, 42228.59414928655, 42315.48763012314, 42222.3153899755, 42156.419412049436, 48529.91872321692, 59869.773791449385]
c= [3193.9016759210845, 3193.21664745992, 3191.7811535690416, 3191.9622649077883, 3189.2239195747297, 3192.86289146058, 3194.8748007053614, 3195.128125585853]
d= [487125.0, 487055.0, 487231.0, 486459.0, 486655.0, 487342.0, 487440.0, 463415.0]

# google
a2= [1107.221115126558, 3464.059368951627, 4451.782527830424, 6398.749739877075, 4740.353161733941, 5980.178981683713, 55035.440331231344, 55533.72202132141]
b2= [49326.27185609308, 48202.84695286925, 47743.50320906361, 48049.7374461297, 47502.67068059332, 47785.65069844525, 54876.18879252564, 62850.16016311581]
c2= [3592.37611260079, 3591.352762434016, 3592.0431267736117, 3590.3328737678385, 3589.7167850399837, 3590.8242576698267, 3594.2459774095837, 3594.4151672891044]
d2= [548666.0, 549911.0, 550158.0, 549530.0, 549243.0, 549786.0, 550164.0, 522895.0]
"""

a= [0.0, 13.65567320127923, 17.780288659611443, 60.78524735630093, 113.7852617154693, 1837.6882888273492, 9803.922836202031, 9827.109086000179]
b= [2907.9785923076643, 2858.774796376898, 2865.2374338860154, 2831.114620458857, 2760.14774545756, 2773.873867263616, 3377.9493997439827, 3441.9705467508747]
c= [61717.0, 61719.0, 61718.0, 61717.0, 61719.0, 61720.0, 61712.0, 61714.0]
d= [61764.0, 61764.0, 61764.0, 61764.0, 61764.0, 61764.0, 61764.0, 61764.0]



alpha = [0.3, 0.5, 0.7, 0.8, 0.9, 1 ,  1, 1]

a1, b1, c1, d1 = sun([a], [b], [c], [d], [5])

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