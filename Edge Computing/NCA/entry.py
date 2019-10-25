import math
import numpy as np
import random
import copy
from Offloading import Offloading
from run import test
import matplotlib.pyplot as plt


run = 11
total_run = 11
ec_o = np.zeros(total_run + 1)
ec_l = np.zeros(total_run + 1)
ec_i = np.zeros(total_run + 1)

while run <= total_run:
    r = Offloading(W=1 * math.pow(10, 6), edge_cpu=25, e=math.pow(10, -9), g=1, number_of_user=15,
                   number_of_edge=1)
    # 0.8347848544951842 0.8380930536576283
    # 0 = local,  1 = full, 2 = partial
    r.full_offload = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    """
    r.D_n = np.array([0.6081713193800378, 0.9629848168985293, 0.6388798080903892, 0.8698480283415665, 0.8983193509302252,
                    0.6038057118654215, 0.989854510772451, 0.5015522289487879, 0.5615684406132868, 0.9778149316032207,
                    0.528483043472088, 0.6306813875839115, 0.9792512708167158, 0.9875185154479058, 0.7555086712802124])
    r.X_n = np.array(
        [378544000, 450088000, 405408000, 224104000, 186000000, 213840000, 397440000, 358080000, 229200000, 408240000,
         450000000, 443424000, 211088000, 348480000, 199144000]).astype(float)
    r.Y_n = np.array(
        [38080000, 68736000, 51200000, 68672000, 77520000, 32448000, 32184000, 78824000, 32936000, 36192000, 110880000,
         38272000, 48216000, 65664000, 38976000]).astype(float)
    r.f_n = np.array([785822799.1649126, 1049455497.7204524, 1010611563.6158786, 1308610580.8006477, 819002730.8090328,
                    1301168449.5903726, 738161974.5994451, 1318882456.9256296, 1323104703.1605594, 1201102948.1874602,
                    623877874.9487828, 1467332175.5420337, 1301672239.1543636, 641621659.4113429, 920079556.1878166])
    r.P_max = np.array([0.8619354565026924, 0.6873828350682971, 0.6360122821755303, 0.5540517562986288, 0.699985309902694,
                      0.6882454769964571, 0.8462063727637572, 0.6034951259552694, 0.7182963636242563,
                      0.8648175333109251, 0.532948223085215, 0.8892560705233667, 0.8077594268704245, 0.5638689400408405,
                      0.5263784986745463])
    r.A = np.array([10520000.0, 11432000.0, 7952000.0, 8152000.0, 7960000.0, 9456000.0, 9464000.0, 11104000.0, 11816000.0,
                  7624000.0, 11344000.0, 9352000.0, 11192000.0, 11880000.0, 8720000.0]) /1.3
    r.B = np.array(
        [3312000.0, 6672000.0, 3512000.0, 7536000.0, 3544000.0, 3584000.0, 3856000.0, 6536000.0, 6912000.0, 7304000.0,
         3424000.0, 6960000.0, 5104000.0, 7616000.0, 5784000.0])

    r.H = np.array([[0.0009548537654942049], [0.0008515731767699408], [0.0007237692955232623], [0.00022466090167174596],
                  [0.001228553612600374], [0.0008590953582013139], [0.0009799935842716486], [0.0018378227924332633],
                  [0.0007861919775785015], [0.0007616707690381632], [0.0007104696641383283], [0.0011748854265119255],
                  [0.0009807788016024898], [0.0013836653575848776], [0.0007547287469046274]])
    """
    """
    r.D_n = np.array(
        [0.9678625688161533, 0.9391072908337577, 0.7160575105250758, 0.7422362609730929, 0.7307333469453672,
         0.7052268214538093, 0.9776858171830731, 0.7696636673416282, 0.854561566962202, 0.6175818057815363,
         0.6653097540511559, 0.634307059296388, 0.796475075887777, 0.8979904695972541, 0.7059191507428056])
    r.X_n = np.array(
        [248192000, 332320000, 336024000, 378768000, 866648000, 555560000, 331336000, 197248000, 259080000, 469224000,
         558912000, 262656000, 247008000, 152160000, 239776000])/1.2
    r.Y_n = np.array(
        [269640000, 233856000, 266760000, 133136000, 223176000, 245784000, 47600000, 88800000, 43632000, 114480000,
         113904000, 162400000, 203520000, 112000000, 222200000])/1.2

    r.f_n = np.array([1190469591.5986264, 934483155.1035804, 1346620919.7615292, 1365350417.2487497, 1158672023.2203248,
                      676925740.9812381, 1407375501.8475528, 1057261073.8485113, 1411089788.2780824, 507445045.69232273,
                      783029522.060963, 827532447.8991139, 1492263137.8169873, 1014892766.5278771, 1000129219.219684])
    r.P_max = np.array(
        [0.9449865559772678, 0.812764200724306, 0.560949660028341, 0.9891341027987275, 0.5329166108012897,
         0.5091786949510437, 0.7878862556273971, 0.7348183937608281, 0.8188553216210156, 0.9437136633275774,
         0.839874491867376, 0.5561487505717277, 0.7399920728680969, 0.8316575259024885, 0.5554677606082268])

    r.A = np.array(
        [7870769.23076923, 6030769.230769231, 7987692.307692307, 8640000.0, 7821538.461538461, 7335384.615384615,
         5378461.538461538, 5507692.307692307, 7926153.846153846, 9003076.923076922, 8695384.615384616,
         6363076.923076923, 4990769.230769231, 8541538.461538462, 6744615.384615384])

    r.B = np.array([5070769.230769231, 5772307.692307692, 4135384.615384615, 3963076.923076923, 2996923.076923077,
                    5563076.923076923, 4621538.461538461, 4873846.153846154, 2756923.076923077, 3698461.5384615385,
                    4775384.615384615, 4523076.923076923, 4812307.692307692, 2873846.1538461535,
                    3630769.2307692305])

    r.H = np.array(
        [([0.00169833]), ([0.00047494]), ([0.00065141]), ([0.00090427]), ([0.0005083]),
         ([0.00051145]), ([0.00077419]), ([0.00044452]), ([0.00045649]), ([0.0007874]),
         ([0.00142386]), ([4.90447803e-04]), ([0.00140683]), ([0.00055599]), ([0.00184667])])
    """
    for n in range(r.number_of_user):
        f_opt = min((r.X_n[n] + r.Y_n[n]) / r.D_n[n], r.f_n[n])
        e_n = r.g * r.k * (r.X_n[n] + r.Y_n[n]) * math.pow(f_opt, 2)
        r.loc_only_e[n] = e_n

    # full 0.012  0.006
    g = int((27 + run * 5)/r.number_of_user)
    r.set_initial_sub_channel(15 * 7, 5, chs=None)
    r.set_multipliers(step=0.000001, p_adjust=1.5, v_n=0, var_k=math.pow(10, -10), delta_l_n=math.pow(10, -13),
                      delta_var_k=math.pow(10, -18), delta_d_n_k=math.pow(10, -15))
    r.set_initial_values()
    ee = r.run(run, t=1, t_max=5000, t_delay=3000, t_stable=1500)
    ec_o[run], ec_l[run], ec_i[run] = test(r)
    run = run + 1

print(ec_o)
print(ec_l)
print("ec_i=", list(ec_i))
print("energy", ee)
print(r.full_offload)
# [6, 6, 10, 6, 13, 6, 8, 7, 10, 10]
plt.plot(ee)
plt.show()