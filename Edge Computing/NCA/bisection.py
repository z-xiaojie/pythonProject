import math
import numpy as np
import random
import matplotlib.pyplot as plt


H = 0.00183782
N = math.pow(10, -9)
G = math.pow(H, 2) / N
d = 4.588262646199881
l_n = 1.9446038700347242e-07

def f(p):
    return 3 * p + d - 3 * l_n * math.pow(10, 6) * math.log2(1 + G * p)

p = 0.001

v = []
t = []
while p <= 0.6034951259552694:
    v.append(f(p))
    t.append(p)
    p += 0.001

plt.plot(t, v)
plt.show()

H = 0.5540517562986288
L = 0
t = 0
while math.fabs(f((H + L)/2)) >= 0.00001:
    print(t, f((H + L)/2), (H + L)/2, H, L)
    if f((H + L)/2) < 0:
        H = (H + L)/2
    else:
        L = (H + L)/2
    t = t + 1

print(">>>>>>>", (H + L)/2)

