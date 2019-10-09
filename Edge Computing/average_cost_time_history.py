from itertools import product
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from mlxtend.plotting import ecdf
from statsmodels.distributions.empirical_distribution import ECDF

e= [np.array([0.57378294, 0.50847061, 0.50375009, 0.50203063, 0.59395578,
       0.52479552, 0.56818893, 0.56585645, 0.60472093, 0.70857181,
       0.49788648, 0.62949892, 0.66991231, 0.528123  , 0.56854998,
       0.60930417, 0.54239629, 0.60062584, 0.59419005, 0.61998929,
       0.67757014, 0.45636302, 0.55913097, 0.51528791, 0.58359108,
       0.58359746, 0.68962661, 0.53960488, 0.62844791, 0.77930981]), np.array([0.56870531, 0.50394373, 0.50002333, 0.49889333, 0.59135548,
       0.52050511, 0.56628492, 0.56315964, 0.59916476, 0.70459331,
       0.49404622, 0.62447941, 0.66354205, 0.52370135, 0.56361681,
       0.60086852, 0.53814288, 0.59628251, 0.58962045, 0.61594625,
       0.67186946, 0.4535517 , 0.55530761, 0.51405928, 0.57739732,
       0.5781315 , 0.69008435, 0.53837957, 0.62755101, 0.77335439]), np.array([0.57044353, 0.50373222, 0.49935394, 0.49874914, 0.59193566,
       0.52162313, 0.56712447, 0.56342943, 0.59713081, 0.70664256,
       0.49398381, 0.62583121, 0.66656042, 0.52355977, 0.56529045,
       0.60247003, 0.53832396, 0.59785818, 0.59055708, 0.61712443,
       0.67436438, 0.45279594, 0.55647303, 0.51235171, 0.57793883,
       0.57881367, 0.6888323 , 0.53655896, 0.62710644, 0.77379012]), np.array([0.56988485, 0.50480344, 0.50108249, 0.49918655, 0.59165207,
       0.52160941, 0.56571771, 0.56285625, 0.59688147, 0.70102107,
       0.49332237, 0.62441586, 0.66360903, 0.52351071, 0.5624226 ,
       0.60191148, 0.53766869, 0.59562153, 0.58896111, 0.61682014,
       0.6715877 , 0.45338482, 0.55471356, 0.51221065, 0.57431917,
       0.57808555, 0.68428327, 0.53622369, 0.6246081 , 0.77333579]), np.array([0.5699834 , 0.50401054, 0.50020664, 0.49865091, 0.59140459,
       0.5199731 , 0.56609349, 0.56244855, 0.59692524, 0.70398028,
       0.49310449, 0.62402406, 0.66329412, 0.52291716, 0.56248701,
       0.60105476, 0.53757267, 0.59690131, 0.58956076, 0.6154769 ,
       0.671119  , 0.45314263, 0.554181  , 0.51314759, 0.57676456,
       0.57674083, 0.68568926, 0.53545778, 0.62373317, 0.76992928]), np.array([0.57245228, 0.5082213 , 0.50321735, 0.50023393, 0.59070914,
       0.5203069 , 0.56567715, 0.56180991, 0.59981531, 0.70115913,
       0.49472953, 0.62479177, 0.66514486, 0.52456021, 0.56257317,
       0.60234841, 0.53800652, 0.5954873 , 0.58745236, 0.6166294 ,
       0.66988739, 0.45300161, 0.55466573, 0.51261015, 0.57622612,
       0.57858877, 0.68159652, 0.53529777, 0.62309773, 0.7722294 ]), np.array([0.56856081, 0.503614  , 0.49957147, 0.49613361, 0.58479152,
       0.51779375, 0.55924911, 0.5577448 , 0.59311102, 0.69432612,
       0.48945693, 0.61901831, 0.65735118, 0.51863459, 0.55646207,
       0.59744921, 0.5330508 , 0.59124445, 0.58444834, 0.60972979,
       0.66450279, 0.45107185, 0.55002066, 0.50699373, 0.57184216,
       0.57150964, 0.66886165, 0.52839961, 0.61542171, 0.7602781 ]), np.array([0.56799626, 0.502971  , 0.49863588, 0.49533895, 0.58126595,
       0.51341696, 0.55543079, 0.55267248, 0.59056644, 0.68766156,
       0.48646553, 0.61400519, 0.65234692, 0.51621131, 0.55096109,
       0.59299638, 0.52899708, 0.58586124, 0.57901303, 0.60411265,
       0.6590212 , 0.44805376, 0.54666964, 0.50367931, 0.56751603,
       0.569777  , 0.66663953, 0.52546154, 0.6123566 , 0.75801408]), np.array([0.56914727, 0.50409589, 0.49987911, 0.4937668 , 0.57921418,
       0.51540497, 0.55372779, 0.55287502, 0.59086003, 0.68649089,
       0.48705573, 0.61377145, 0.65076985, 0.51665619, 0.55167162,
       0.5959539 , 0.52991664, 0.58544232, 0.57968569, 0.60346594,
       0.65940968, 0.44972172, 0.54823106, 0.50110928, 0.56671345,
       0.56811702, 0.65566628, 0.51948142, 0.60653394, 0.7517471 ]), np.array([0.5651122 , 0.49619573, 0.49021433, 0.48372782, 0.5639164 ,
       0.50356301, 0.53916033, 0.53882742, 0.57822294, 0.67130633,
       0.47394692, 0.59770127, 0.63450413, 0.50281875, 0.53435243,
       0.57913547, 0.51584938, 0.57016495, 0.56376621, 0.58618511,
       0.64109278, 0.43702023, 0.53434658, 0.48795956, 0.55351292,
       0.55622885, 0.64202617, 0.50884736, 0.59576137, 0.73772327]), np.array([0.55334574, 0.48938381, 0.48329819, 0.47944659, 0.55948924,
       0.50012044, 0.53455366, 0.53361477, 0.57497459, 0.66400254,
       0.46950329, 0.59342516, 0.6283339 , 0.49837574, 0.52817691,
       0.57385281, 0.51022829, 0.56365653, 0.55757261, 0.58189881,
       0.63331915, 0.4330597 , 0.52902369, 0.4842021 , 0.54828355,
       0.55151054, 0.63520393, 0.50570176, 0.58889793, 0.73111779]), np.array([0.68317379, 0.60865129, 0.59996279, 0.59621698, 0.69996237,
       0.6226155 , 0.67119653, 0.67969095, 0.7141621 , 0.81422279,
       0.59963739, 0.72792295, 0.77641225, 0.62297374, 0.66877074,
       0.70708292, 0.63441185, 0.71265947, 0.69686092, 0.72227913,
       0.79272348, 0.54659129, 0.6572392 , 0.60769628, 0.68687277,
       0.66320162, 0.81247566, 0.63880299, 0.71750659, 0.89101023]), np.array([0.68638976, 0.61284611, 0.59865627, 0.58886389, 0.69873783,
       0.626415  , 0.67962952, 0.67236487, 0.71593315, 0.82567067,
       0.60240423, 0.73459811, 0.78731174, 0.62593652, 0.67682322,
       0.71043164, 0.6354413 , 0.70463963, 0.70086416, 0.72545867,
       0.79426332, 0.54748845, 0.67147604, 0.61799126, 0.68623964,
       0.67224838, 0.81470492, 0.6343927 , 0.71936357, 0.88571436])]

alpha  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1, 1, 1]

for i in range(len(alpha)):
    print("a"+str(alpha[i])+"=", list(e[i]/1))
