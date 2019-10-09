import math


def total_utility_density(users):
    total_utility = 0
    total_density = 0
    for n in range(len(users)):
        total_utility += users[n].get_utility_with_rate()
        total_density += users[n].get_density_with_rate()
    return total_utility, total_density


def feasible2(users, m):
    for t in range(10000):
        dbf_sum = 0
        for u in users:
            dbf_sum += dbf(u, t)
            if dbf_sum > m*t:
                return False
    return True


def dbf(user, t):
    return max(0, (math.floor((t - user.get_deadline_with_rate()) / max(user.interval, user.job_size / user.rate)) + 1)) * user.exe_time
    #if t < user.actual_deadline:
        #return 0
    #else:
        #return user.exe_time + user.get_utility() * (t - user.actual_deadline)


def rbf(user, t):
    return user.exe_time + user.get_utility() * t


def condition_one(user, users):
    dbf_sum = 0
    for u in users:
        dbf_sum += dbf(u, user.get_deadline_with_rate())
    return (user.get_deadline_with_rate() - dbf_sum) >= user.exe_time


def condition_two(user, users):
    utility_sum = 0
    for u in users:
        utility_sum += u.get_utility_with_rate()
    density_sum = 0
    for u in users:
        density_sum += u.get_density_with_rate()
    return (1 - utility_sum) >= user.get_utility_with_rate() and (1 - density_sum) >= user.get_density_with_rate()


def check_job_fail_summary(edge, number_of_edge):
    sum_fail = 0
    for k in range(number_of_edge):
        if edge[k].number_of_job > 0:
            sum_fail += edge[k].number_of_failed/edge[k].number_of_job
    return sum_fail/number_of_edge

