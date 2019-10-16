import matplotlib.pyplot as plt
import numpy as np
import math


D_n=np.array( [0.6081713193800378, 0.9629848168985293, 0.6388798080903892, 0.8698480283415665, 0.8983193509302252, 0.6038057118654215, 0.989854510772451, 0.5015522289487879, 0.5615684406132868, 0.9778149316032207, 0.528483043472088, 0.6306813875839115, 0.9792512708167158, 0.9875185154479058, 0.7555086712802124] )
X_n=np.array( [378544000, 450088000, 405408000, 224104000, 186000000, 213840000, 397440000, 358080000, 229200000, 408240000, 450000000, 443424000, 211088000, 348480000, 199144000] ).astype(float)
Y_n=np.array( [38080000, 68736000, 51200000, 68672000, 77520000, 32448000, 32184000, 78824000, 32936000, 36192000, 110880000, 38272000, 48216000, 65664000, 38976000] ).astype(float)
f_n=np.array( [785822799.1649126, 1049455497.7204524, 1010611563.6158786, 1308610580.8006477, 819002730.8090328, 1301168449.5903726, 738161974.5994451, 1318882456.9256296, 1323104703.1605594, 1201102948.1874602, 623877874.9487828, 1467332175.5420337, 1301672239.1543636, 641621659.4113429, 920079556.1878166] )
P_max=np.array( [0.8619354565026924, 0.6873828350682971, 0.6360122821755303, 0.5540517562986288, 0.699985309902694, 0.6882454769964571, 0.8462063727637572, 0.6034951259552694, 0.7182963636242563, 0.8648175333109251, 0.532948223085215, 0.8892560705233667, 0.8077594268704245, 0.5638689400408405, 0.5263784986745463] )
A=np.array( [10520000.0, 11432000.0, 7952000.0, 8152000.0, 7960000.0, 9456000.0, 9464000.0, 11104000.0, 11816000.0, 7624000.0, 11344000.0, 9352000.0, 11192000.0, 11880000.0, 8720000.0] )/1.3
B=np.array( [3312000.0, 6672000.0, 3512000.0, 7536000.0, 3544000.0, 3584000.0, 3856000.0, 6536000.0, 6912000.0, 7304000.0, 3424000.0, 6960000.0, 5104000.0, 7616000.0, 5784000.0] )
H=np.array( [0.0009548537654942049, 0.0008515731767699408, 0.0007237692955232623, 0.00022466090167174596, 0.001228553612600374, 0.0008590953582013139, 0.0009799935842716486, 0.0018378227924332633, 0.0007861919775785015, 0.0007616707690381632, 0.0007104696641383283, 0.0011748854265119255, 0.0009807788016024898, 0.0013836653575848776, 0.0007547287469046274] )

cpu = [1.3945, 1.3963, 1.6525, 1.3012, 0.6554, 0.9383, 0.8878, 2.1928, 1.4859, 1.1201, 2.7356, 1.6949, 0.6321, 1.021, 0.8917]
channels = [4, 5, 4, 13, 3, 4, 3, 7, 6, 6, 6, 7, 4, 5, 5]

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