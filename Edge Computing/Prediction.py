import numpy as np


########################
#  average bandwidth predicted that each edge can use
########################
def assign_channel_to_edge(raw_data, time_interval=33, ch=11):
    channel = np.array(raw_data[ch])
    time_max = len(channel)
    avg = []
    std = []
    t = 0
    time_interval = int(time_interval/1.8)
    while t + time_interval < time_max:
        tp = np.sum(channel[t:t + time_interval]) / time_interval
        avg.append(tp)
        std.append(np.std(channel[t:t + time_interval]))
        t = t + time_interval
    return avg

