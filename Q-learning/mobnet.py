import numpy as np
import scipy.io as sio
import random


def statistic(data_source, mean, frequency_selected, CH_state):
    mat_contents = sio.loadmat(data_source + ".mat")
    measurement_results = np.array(mat_contents["measurementResults"])
    for sweep in range(measurement_results.shape[0]):
        for i in range(len(frequency_selected)):
            if measurement_results[sweep][frequency_selected[i]] > mean:
                CH_state[i].append(1)
            else:
                CH_state[i].append(0)

def create_channel(number_ch, threshold=-107, size = 150):
    # start_f = np.random.randint(low=300, high=8000)
    frequency_selected = random.sample(range(0, 8192), number_ch)
    CH_power = []
    CH_state = []
    for i in range(number_ch):
        CH_power.append([])
        CH_state.append([])
    print("create ", number_ch, "channels, with frequency ", frequency_selected)
    for i in range(1, size, 1):
        if i < 10:
            statistic("data/02_NE/0770MHz/MeasRes_0770_000" + str(i), threshold, frequency_selected, CH_state)
        elif 10 <= i < 100:
            statistic("data/02_NE/0770MHz/MeasRes_0770_00" + str(i), threshold, frequency_selected, CH_state)
        else:
            statistic("data/02_NE/0770MHz/MeasRes_0770_0" + str(i), threshold, frequency_selected, CH_state)
        print("done " + str(i))
    CH_power = np.asarray(CH_power).astype(int)
    CH_state = np.asarray(CH_state).astype(int)
    np.savetxt("train_power.csv", np.transpose(CH_power), delimiter=",", fmt='%i')
    np.savetxt("train_state.csv", np.transpose(CH_state), delimiter=",", fmt='%i')
    return number_ch
