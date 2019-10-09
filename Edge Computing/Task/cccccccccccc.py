from itertools import product
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from mlxtend.plotting import ecdf
from statsmodels.distributions.empirical_distribution import ECDF


def show(a, b, c, label):
    plt.subplot(221)
    plt.plot(alpha, a, marker="X", label=label)

    plt.subplot(222)
    plt.plot(alpha, b, marker="+", label=label)

    plt.subplot(223)
    plt.plot(alpha, c, marker="+", label=label)


def cal(a , b, c, number_of_run):
    for k in range(len(alpha)):
        a[k] = a[k] / number_of_run
        b[k] = b[k] / number_of_run
        c[k] = c[k] / number_of_run
    return a, b, c


def sun(a1, b1, c1, num):
    a = np.zeros(len(alpha))
    b = np.zeros(len(alpha))
    c = np.zeros(len(alpha))
    for i in range(len(num)):
        for k in range(len(alpha)):
            a[k] += a1[i][k] / num[i]
            b[k] += b1[i][k] / num[i]
            c[k] += c1[i][k] / num[i]
    a = a / len(num)
    b = b / len(num)
    c = c / len(num)
    return a, b, c


a1 = [15.037919639133952, 16.108824551124883, 20.58764696606601, 19.123208135689612, 25.687191099711463, 30.831449301794766, 35.71361461325748, 48.27397820063869, 59.501474176227205, 85.38477645404836, 346.8455600672897]
b1 = [1276.4862220664068, 1253.6058732822107, 1233.7926298846417, 1233.0375538308476, 1237.08899359775, 1192.1105709620535, 1168.753305377551, 1153.2396401626006, 1148.5944952986385, 1139.0662125318502, 1150.977483413994]
c1 = [99.80669427388774, 99.84937257890971, 99.75538472199909, 99.8431283707915, 99.8422149613831, 99.79301425259402, 99.84227506254145, 99.83037737258098, 99.77766357289156, 99.83426094120571, 99.82955745499764]

# from google High Migration Cost
a3= [27.840953812755515, 28.481346799843884, 29.148933626013395, 30.39831820019653, 26.81263715166265, 30.053257982058007, 50.95923156288031, 56.974510472912996, 107.47464529819233, 94.6349402269288, 253.90178168489638]
b3= [1231.9160752319458, 1228.0238424104489, 1207.3328361329775, 1232.2055083793857, 1215.5071943172945, 1205.5927627464162, 1170.9461021317015, 1174.194565681289, 1159.2008377506022, 1151.0226384000048, 1141.7820807672015]
c3= [99.73489875111657, 99.8206377561046, 99.81803132267058, 99.843921102924, 99.82565245839237, 99.80497253939177, 99.84350300932894, 99.84147195342555, 99.8124459748977, 99.82068189247076, 99.81477345925643]


# Low Migration Cost
a2 = [7.339761701889401, 8.739209428225406, 6.7062543305767095, 11.804698098126327, 12.161475198747741, 22.179394608050284, 26.256667729409457, 24.825315847096405, 37.60411979914028, 55.88466973088039, 145.44939126543153]
b2 = [1261.52655065367, 1232.0350814811704, 1205.9311121154587, 1191.5606680603994, 1154.9445702919052, 1152.1457427423045, 1148.1432117560628, 1123.018956715667, 1129.1704288954884, 1129.1512924827448, 1123.803851082671]
c2 = [99.85107091568135, 99.78578349488002, 99.83856525441772, 99.84324288304973, 99.80756900074311, 99.83233539308016, 99.82791266357661, 99.83485681010191, 99.8135320511162, 99.82920459790843, 99.83237218216863]

# Low Migration Cost 50 runs
a4= [14.316859618746225, 12.608599293101586, 12.381988437398912, 15.380229247805564, 26.058956857016796, 24.834061821287424, 25.704491279269334, 48.35788010437065, 67.53011575342174, 75.10104078898259, 144.60458559156078]
b4= [1257.791715852643, 1221.4366243826753, 1203.5999607317774, 1186.4171077227777, 1177.5983431354343, 1167.179224156771, 1160.2088262053883, 1154.6307375908987, 1158.902566343768, 1148.0586125502718, 1147.6834438829628]
c4= [99.8252035589298, 99.81907361743981, 99.81421109244913, 99.79880293652295, 99.82830085284532, 99.80944542753974, 99.81292408806597, 99.83059486702676, 99.81916814503882, 99.79022882274667, 99.82424871686237]

alpha = np.arange(0, 1.1, 0.1)


"""
for k in range(len(alpha)):
    a[k] = a1[k] + a2[k] / number_of_run
    b[k] = b1[k] + b2[k] / number_of_run
    c[k] = c1[k] + c2[k] / number_of_run
"""

a1, b1, c1 = sun([a1,a3, a2,a4],[b1, b3, b2,b4],[c1,c3,c2,c4] , [1, 1,1,1])

print(alpha)

#a1, b1, c1 = sun([a4],[b4],[c4], [1])

print(list(a1))
print(list(b1))
print(list(c1))

#a3, b3, c3 = cal(a3 , b3, c3, 15)

#a4, b4, c4 = cal(a4 , b4, c4, 33)

#show(a3, b3, c3, "High Migration Cost")
#show(a3, b3, c3, "High Migration Cost")
#show(a4, b4, c4, "Low Migration Cost")
show(a1, b1, c1, "Low Migration Cost")


plt.subplot(221)
plt.xlabel("alpha")
plt.ylabel("Migration Overhead [s]")
plt.legend()
plt.xticks(alpha)
plt.xlim([0, np.max(alpha)])
plt.grid(True)

plt.subplot(222)
plt.xlabel("alpha")
plt.ylabel("TF/JOB [ms]")
plt.legend()
plt.xticks(alpha)
plt.xlim([0, np.max(alpha)])
plt.grid(True)

plt.subplot(223)
plt.xlabel("alpha")
plt.ylabel("Job Success [%]")
plt.legend()
plt.xlim([0, np.max(alpha)])
plt.grid(True)


plt.tight_layout()
plt.show()