import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def temp_summary(start, end, t_max, data, agents, dimension):
    total = 0
    while start <= end and start < t_max:
        valid = 0
        for col in range(dimension):
            if data[col][start] == 0:
                valid += 1
        if valid > agents:
            valid = agents
        total += valid
        start += 1
    return round(total, 3)


data = []
ch_state = pd.read_csv("train_state.csv", header=None)
for i in range(len(ch_state.columns)):
    data.append(ch_state[ch_state.columns[i]])
t_max = len(data[0])
t = 0
total = []
while t < t_max:
    tp = temp_summary(t, t+1000, t_max, data, 10, 14)
    total.append(tp)
    t = t + 1000
    print(t, tp)
plt.plot([0.955, 0.957, 0.957, 0.953, 0.947, 0.941, 0.933, 0.929, 0.922, 0.916, 0.91, 0.899, 0.892, 0.888, 0.883, 0.877, 0.865, 0.86, 0.858, 0.857, 0.858, 0.857, 0.856, 0.856, 0.858, 0.86, 0.863, 0.866, 0.868, 0.871, 0.874, 0.876, 0.879, 0.881, 0.883, 0.885, 0.887, 0.889, 0.89, 0.891, 0.892, 0.892, 0.891, 0.892, 0.893, 0.895, 0.896, 0.897, 0.898, 0.899, 0.899, 0.897, 0.896, 0.894, 0.894, 0.893, 0.891, 0.89, 0.889, 0.888, 0.886, 0.885, 0.883, 0.882, 0.879, 0.876, 0.873, 0.871, 0.869, 0.866, 0.864, 0.862, 0.861, 0.861, 0.861, 0.86, 0.858, 0.856, 0.854, 0.851, 0.849, 0.849, 0.85, 0.85, 0.851])
plt.show()