import math
import numpy as np
import random


number_of_user = 5
number_of_edge = 3

edge_cpu = np.array([random.randint(10, 20) * math.pow(10, 9) for x in range(number_of_edge)])
user_cpu = np.array([random.uniform(1, 2) * math.pow(10, 9) for x in range(number_of_user)])

D_n = np.array([random.uniform(0.5, 1) for x in range(number_of_user)])
D_n_k = np.array([[D_n[n] - random.uniform(0.05, 0.15) for k in range(number_of_edge)] for n in range(number_of_user)])

Y_n = np.array([0 for x in range(number_of_user)])
for n in range(number_of_user):
    y_max = np.mean(D_n_k[n]) * user_cpu[n]
    y = random.randint(250, 500) * 8000 * random.randint(50, 250)
    Y_n[n] = min(y_max, y)

# Y_n = np.array([random.randint(250, 500) * 8000 * random.randint(50, 250) for x in range(number_of_user)])

X_n = np.array([random.randint(250, 500) * 8000 * random.randint(50, 250) for x in range(number_of_user)])



f_n = np.array([[user_cpu[n] for k in range(number_of_edge)] for n in range(number_of_user)])

g = 0.9

th_k = np.array([edge_cpu[k]/(40*number_of_user) for k in range(number_of_edge)])


K = math.pow(10, -27)

c2 = np.array([1.0 for k in range(number_of_edge)])
c1 = np.array([[user_cpu[n] * user_cpu[n] * user_cpu[n] * 2 * g * K for k in range(number_of_edge)] for n in range(number_of_user)])
f_n_k = np.array([[ (c2[k] * c1[n][k] / (2 * (1 - g) * K ))**(1./3.) for k in range(number_of_edge)] for n in range(number_of_user)])


print([(X_n[n] + Y_n[n]) / (D_n[n] * math.pow(10, 9)) for n in range(number_of_user)])

print(f_n / math.pow(10, 9))
print(f_n_k / math.pow(10, 9))

step = 0.1


# find optimal a_n_k
def assignment(f_n, f_n_k):
    print(">>>>>>>>>>>>assignment>>>>>>>>>>>>")
    # initialize a_n_k
    a_n_k = np.array([[0 for y in range(number_of_edge)] for n in range(number_of_user)])
    E_n_k = np.array([[1.0 for y in range(number_of_edge)] for n in range(number_of_user)])
    E_n = np.array([1.0 for n in range(number_of_user)])
    for n in range(number_of_user):
        opt_k = -1
        opt_diff = 9999
        for k in range(number_of_edge):
            E_n_k[n][k] = g * K * Y_n[n] * math.pow(f_n[n][k], 2) + (1 - g) * K * X_n[n] * math.pow(f_n_k[n][k], 2)
            # local optimal f = z / d
            f_opt = min((X_n[n] + Y_n[n])/D_n[n], user_cpu[n])
            #print(E_n_k, E_n)
            if opt_k == -1 and (D_n_k[n][k] - X_n[n] / f_n_k[n][k] - Y_n[n] / f_n[n][k]) >= 0:
                opt_k = k
                opt_diff = E_n_k[n][k] - E_n[n]
            else:
                if opt_diff > (E_n_k[n][k] - E_n[n]) and (D_n_k[n][k] - X_n[n] / f_n_k[n][k] - Y_n[n] / f_n[n][k]) >= 0:
                    opt_k = k
                    opt_diff = E_n_k[n][k] - E_n[n]
        E_n[n] = g * K * (X_n[n] + Y_n[n]) * math.pow(f_opt, 2)
        if opt_k != -1 and (E_n_k[n][k] <= E_n[n] or (X_n[n] + Y_n[n])/D_n[n] > user_cpu[n]):
            a_n_k[n][opt_k] = 1

    return a_n_k, E_n_k, E_n


def update(t, c1, c2, f_n, f_n_k):
    print(">>>>>>>>>>>>update>>>>>>>>>>>>")
    stop = True
    e = 0.005
    for n in range(number_of_user):
        for k in range(number_of_edge):
            new_c1 = max(0, c1[n][k] + math.sqrt(1/t) * (X_n[n]/f_n_k[n][k] + Y_n[n]/f_n[n][k] - D_n_k[n][k]))
            if abs(new_c1 - c1[n][k]) > e:
                print("e:", abs(new_c1 - c1[n][k]))
                c1[n][k] = new_c1
                stop = False

    for k in range(number_of_edge):
        d_c2 = 0
        for n in range(number_of_user):
            d_c2 += f_n_k[n][k]
        new_c2 = max(0, c2[k] - math.sqrt(1/t) * (d_c2/edge_cpu[k] - 1))
        if math.fabs(new_c2 - c2[k]) > e:
            #print(math.fabs(new_c2 - c2[k]))
            c2[k] = new_c2
            stop = False


    #print("c1",c1)
    #print("c3", c3)
    #print("c2", c2)
    return stop

def new(c1, c2):
    #f_n_k_p = np.array([[ math.sqrt(c1[n][k] * X_n[n] / (c3[n][k] + c2[k])) - th_k[k] for k in range(number_of_edge)] for n in range(number_of_user)])
    f_n = np.array([[ (c1[n][k] / (2 * g * K))**(1./3.) for k in range(number_of_edge)] for n in range(number_of_user)])
    f_n_k = np.array([[(c2[k] * c1[n][k] / (2 * (1 - g) * K)) ** (1. / 3.) for k in range(number_of_edge)] for n in range(number_of_user)])

    for n in range(number_of_user):
        for k in range(number_of_edge):
            if f_n[n][k] > user_cpu[n]:
                f_n[n][k] = user_cpu[n]

    print(f_n / math.pow(10, 9))
    print(f_n_k / math.pow(10, 9))

    return f_n, f_n_k

t = 1

while t <= 10000:
    a_n_k, E_n_k, E_n = assignment(f_n, f_n_k)
    stop1 = update(t, c1, c2, f_n, f_n_k)
    f_n, f_n_k = new(c1, c2)
    t = t + 1

    F = np.array([0 for n in range(number_of_user)])
    for n in range(number_of_user):
        for k in range(number_of_edge):
            if a_n_k[n][k] == 1 and (D_n_k[n][k] - X_n[n] / f_n_k[n][k] - Y_n[n] / f_n[n][k]) >= 0:
                F[n] = 1
        f_opt = min((X_n[n] + Y_n[n]) / D_n[n], user_cpu[n])
        if F[n] == 0 and (D_n[n] - (X_n[n] + Y_n[n]) / f_opt) >= 0:
            F[n] = 1

    if stop1:
        break

print([(X_n[n] + Y_n[n]) / (D_n[n] * math.pow(10, 9)) for n in range(number_of_user)])
print(a_n_k)
print(edge_cpu / math.pow(10, 9))
print(user_cpu / math.pow(10, 9))
print("X", X_n / math.pow(10, 9))
print("Y", Y_n / math.pow(10, 9))
print("deadline")
print(D_n)
print("deadline")
print(D_n_k)
print("energy")
print(E_n_k)
print("energy")
print(E_n)

ee = []
for n in range(number_of_user):
    F = False
    for k in range(number_of_edge):
        if a_n_k[n][k] == 1:
            ee.append(E_n_k[n][k])
            print("completion", D_n_k[n][k] - X_n[n]/f_n_k[n][k] - Y_n[n]/f_n[n][k])
            F = True
    if F == False:
        ee.append(E_n[n])
        f_opt = min((X_n[n] + Y_n[n]) / D_n[n], user_cpu[n])
        print("completion", D_n[n] - (X_n[n] + Y_n[n]) / f_opt )

print("Total Energy", np.sum(ee))
print("Total Energy via task offloading", np.sum(E_n))


