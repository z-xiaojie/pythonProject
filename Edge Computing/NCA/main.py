from ML.offload_2 import Offloading
from matplotlib import pyplot as plt
import numpy as np


L = 0

y = np.zeros(L + 1)
z = np.zeros(L + 1)
f = np.zeros(L + 1)
run = 0
while run <= 0:
    i = 0
    number_of_user = 10
    while i <= L:
        r = Offloading(step=0.0001, e=0.00005, g=1, number_of_user=number_of_user, number_of_edge=1)
        r1, r2, r3 = r.run(t=1, t_max=30000)
        y[i] = y[i] + r1
        z[i] = z[i] + r2
        f[i] = f[i] + r3
        number_of_user += 5
        i += 1
    run = run + 1

plt.plot(y/6, label="offload")
plt.plot(z/6, label="local only")
plt.legend()
#plt.show()

print(f)