import matplotlib.pyplot as plt
import numpy as np
import math


D_n = np.array(
        [0.6434760449620666, 0.6985855701819832, 0.5953202841374898, 0.6720251404961805, 0.9206210957361978,
         0.9940730855775903, 0.8470228512001519, 0.5756948858769262, 0.9678395112162889, 0.5188857824056006,
         0.8678558972843435, 0.5816600566603978, 0.9776131250172432, 0.9201693324099356, 0.8321262496463979])
X_n = np.array(
        [167232000, 319232000, 359856000, 446160000, 900176000, 165432000, 243648000, 650496000, 209328000, 332664000,
         308256000, 262752000, 174592000, 581808000, 678000000])
Y_n = np.array(
        [210560000, 76464000, 264096000, 179424000, 129336000, 205216000, 75504000, 118320000, 88736000, 60168000,
         299200000, 68608000, 223080000, 126768000, 83448000])
user_cpu = np.array(
        [1297771613.017809, 618001428.8067482, 1125608844.8521261, 838053481.4870487, 1213260766.3677754,
         719413447.7364688, 1115691295.7945516, 1227277218.1211252, 764604954.8334502, 934467049.53918,
         834192580.1475035, 1148056626.3936508, 1062324656.3660879, 969174254.3867567, 1045136843.9318776])
P_max = np.array([0.7682643064440966, 0.7358605028076793, 0.51642734214858, 0.693251432100195, 0.6657518875531107,
                        0.5505576230349652, 0.8480135128628967, 0.8233722016895986, 0.902649740396219,
                        0.5533851106324029, 0.9693683580362833, 0.9941409984422032, 0.582016145105133,
                        0.9293443212945207, 0.5039365160281808])
A = np.array([9187692.307692308, 9150769.23076923, 6966153.846153846, 6196923.076923077, 6172307.692307692,
                    6781538.461538461, 9113846.153846154, 6886153.846153846, 4732307.692307692, 4627692.307692308,
                    6935384.615384615, 5083076.923076923, 6000000.0, 5144615.384615384, 8744615.384615384])
B = np.array([4073846.1538461535, 4326153.846153846, 3858461.5384615385, 2430769.2307692305, 5458461.538461538,
                    2486153.846153846, 5126153.846153846, 2701538.4615384615, 3089230.769230769, 5778461.538461538,
                    3378461.5384615385, 4190769.2307692305, 5101538.461538461, 2990769.2307692305, 4935384.615384615])

H = np.array(
        [([0.00233117]), ([0.0008412]), ([0.0003165]), ([0.00049224]), ([0.00056272]),
         ([0.0016347]), ([0.00167008]), ([0.00095469]), ([0.00098668]), ([0.00027142]),
         ([0.00097651]), ([0.00138947]), ([0.00126355]), ([0.00087926]), ([0.00052146])])

H = H.reshape(-1)

priority = []
gain = []
value = []
for n in range(15):
    a = X_n[n] / np.sum(X_n)
    b = B[n]/np.sum(B)
    value.append(a / b)
    print(a, b)
    G = (P_max[n]) * math.pow(H[n], 2)/math.pow(10, -9)
    R = math.pow(10, 6) * math.log2(1 + G)
    l_n = (math.pow(10, -9)/math.pow(H[n], 2) + P_max[n]) * (1 + 0) * math.log(2) / math.pow(10, 6)
    ch_gain = R - math.pow(10, 6) * G / (math.log(2) * (1 + G))
    gain.append(ch_gain/10000000)
    # [2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 1, 2, 2, 0]

for n in range(15):
    priority.append(value[n] + gain[n])

dist = [[i+1, priority[i]] for i in range(len(priority))]
dist.sort(key=lambda x: x[1], reverse=True)

save = [ 0.91780743 , 0.80197061,  0.88782169 ,-4.01512869 , 0.4295655  , 0.6065478,
  0.85348899  ,0.90149447 , 0.32103407 , 0.71240669 , 0.75768659  ,0.92200596,
  0.20692837 , 0.7531216 , -0.09093574]

dist2 = [[i+1, save[i]] for i in range(len(save))]
dist2.sort(key=lambda x: x[1], reverse=True)

dist3 = [[i+1, gain[i]] for i in range(len(gain))]
dist3.sort(key=lambda x: x[1], reverse=True)

print("gain ", [dist3[i][0] for i in range(len(dist3))])

print("value 1", np.array(value).round(5))
print("gain 1", np.array(gain).round(5))

print("order", [dist[i][0] for i in range(len(dist))])
#print(dist)
print("save ",[ dist2[i][0] for i in range(len(dist2))])

# full_offload = [2,  2,  2, 2,  2,  2,  0,  2,  2,  0]
print(priority)
avg = np.average(priority)
print(avg)
full_offload = np.zeros(15).astype(int) + 2
for n in range(15):
    if priority[n] < avg and math.fabs(priority[n] - avg) > avg * 0.2:
        full_offload[n] = full_offload[n] - 2
print(list(full_offload))



plt.subplot(2, 2, 1)
plt.bar(np.arange(len(A))+1, A/8000, label="Full Offloading")
plt.bar(np.arange(len(B))+1, B/8000, label="Partial Offloading")
plt.xlabel("Task ID")
plt.ylabel("Input Data")
plt.legend()

plt.subplot(2,2,2)
plt.bar(np.arange(len(A))+1, (X_n + Y_n) /math.pow(10, 9), label="Local Computation")
plt.bar(np.arange(len(B))+1, X_n/math.pow(10, 9), label="Edge Computation")
plt.xlabel("Task ID")
plt.ylabel("Process Requirement")
plt.legend()

plt.subplot(2,2,3)
plt.bar(np.arange(len(H))+1, H, label="Sub-channel Gain")
plt.xlabel("Task ID")
plt.ylabel("Sub-channel Gain")
plt.legend()

plt.subplot(2,2,4)
#plt.bar(np.arange(len(H))+1, save, label="Save")
plt.bar(np.arange(len(H))+1, np.array(priority)*3, label="priority")
plt.xlabel("Task ID")
plt.ylabel("Priority")
plt.legend()

print("total computation", X_n/math.pow(10, 9), Y_n/math.pow(10, 9))

plt.show()

# [1, 7, 12, 3, 11, 10, 2, 8, 5, 6, 13, 14, 9, 15, 4]