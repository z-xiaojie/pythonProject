import numpy as np
from Task.Optimization import global_optimization, local_optimization
from Task.Assignment_Map import AssignmentMemory
import time


def compare(run, alpha, marker, label, evn, edge, users, number_of_edge, number_of_user, change=0, verbose=1,
            time_interval=None, system_interval=None, start_time=None, test_duration=None, system_timer=None):
    global partition_history
    ########################
    #  system startup
    ########################
    migration_overhead = 0
    migration_overhead_history = []

    cost_history = []
    cost_total = 0

    avg_cost_history = []
    avg_cost = 0
    avg_number_of_job = 0

    number_of_job = 0
    cur_time = start_time
    user_partition = None

    none_service = 0
    none_service_history = []

    complete_history = []
    complete_history_time = []
    number_of_release_job_time =0
    number_of_release_job = 0

    #
    memory = AssignmentMemory(max_size=100, min_diff=0.05, min_better=number_of_edge)
    past_targeted = 0
    start = time.time()

    for n in range(number_of_user):
        users[n].policy = change


    migration_overhead_fail = 0
    # every iteration is 1
    while cur_time <= start_time + test_duration:
        # time_interval : number_of_failed / number_of_job
        #fail_level = check_job_fail_summary(edge, number_of_edge)

        if cur_time % time_interval == 0 or cur_time == start_time:

            """
                1.   always choose opt
                2.   only opt > 0, do service migration
                3.   only current partition is not feasible, do service migration
                4.   only current partition is not feasible and opt > 0, do service migration
            """

            # proactive migration
            if change >= -1 or cur_time == start_time:

                bandwidth = list()
                for k in range(number_of_edge):
                    bandwidth.append(round(edge[k].avg_data[int(edge[k].cur_time / edge[k].interval)], 3))

                #print(bandwidth)

                past, rf, user_partition,rf_f = global_optimization(alpha, users, edge, number_of_user, number_of_edge,
                                                         user_partition, memory, bandwidth, policy=change)
                migration_overhead += rf
                migration_overhead_fail += rf_f
                past_targeted += past

            scheduling = False
            for k in range(number_of_edge):
                if edge[k].get_number_of_users() == 0:
                    continue
                avg_rate = edge[k].avg_data[int(edge[k].cur_time / edge[k].interval)]
                edge[k].allocate_fair_rate(avg_rate)
                core_density, core_utility = edge[k].summary()
                for density in core_density:
                    if density > 1:
                        scheduling = True
                        break
            if scheduling:
                local_optimization(edge, number_of_edge)

        if cur_time % system_interval == 0 and cur_time > start_time and number_of_job > 0:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("time", cur_time - start_time,"run", run, "alpha", alpha, "spend time=", round((time.time() - start)/3, 2))
            start = time.time()
            #print("user partition", user_partition)
            #print("number of job", number_of_job)
            #print("non service", none_service)
            print("migration_overhead", migration_overhead)
            #print("cur_time", cur_time - start_time)
            #print("past_targeted = ", past_targeted)
            #print("past_assignment = ", len(memory.past_assignment))
            total_number_of_complete = 0
            for k in range(number_of_edge):
                total_number_of_complete += edge[k].total_number_of_complete
            complete_history.append(total_number_of_complete/number_of_release_job)

            #print("job complete", complete_history[-1])
            print("transmission cost", cost_total / number_of_job)
            migration_overhead_history.append(migration_overhead)

            # for time line information
            number_of_complete_time = 0
            number_of_transmitted_time = 0
            for k in range(number_of_edge):
                number_of_complete_time += edge[k].number_of_complete
                number_of_transmitted_time += edge[k].number_of_job
            complete_history_time.append(number_of_complete_time/number_of_transmitted_time)
            if complete_history_time[-1] > 1:
                complete_history_time[-1] = 1
            number_of_release_job_time = 0
            print("job complete", complete_history_time[-1])

            avg_cost_history.append(avg_cost)
            avg_cost = 0
            avg_number_of_job = 0


            """
            none_service_history.append(math.ceil(none_service/system_interval))
            none_service = 0
            """
            for k in range(number_of_edge):
                cloud = edge[k]
                cloud.clear_data()

        # 0.53 15296
        for k in range(number_of_edge):
            cloud = edge[k]
            cloud.exe()
            # real adjust
            adjust = evn[cloud.ch][int(cur_time/1.8)]
            if cloud.get_number_of_users() == 0:
                continue
            else:
                cloud.allocate_fair_rate(adjust)

        #print(user_partition)
        for n in range(number_of_user):
            if user_partition[users[n].id] == -1:
                continue
            users[n].step(cur_time)
            k = edge[user_partition[users[n].id]].id
            m = edge[k].find_core_to_user_by_id(users[n].id)

            released = users[n].release_job(k, m)

            number_of_release_job += released

            # for time line information
            number_of_release_job_time += released
            """
            if m is not None:
                # release job and add to local queue
                users[n].release_job(k, m)
            else:
                none_service += 1
            """

        # timer + 1 s
        cur_time = cur_time + system_timer
        for n in range(number_of_user):
            users[n].step(cur_time)
        for k in range(number_of_edge):
            edge[k].step(cur_time)

        for n in range(number_of_user):
            k = edge[user_partition[n]].id
            m = edge[k].find_core_to_user_by_id(users[n].id)
            if m is None or users[n].rate == 0:
                continue
            # transmit the top job in queue
            status, job, overflow = users[n].transmit_job()
            if status == "Transmitted":
                edge[k].receive_job(job, m)
                job_cost = cur_time - job.release_time - overflow
                number_of_job += 1
                cost_total += job_cost
                avg_cost = (avg_cost * avg_number_of_job + job_cost) / (avg_number_of_job + 1)
                avg_number_of_job = avg_number_of_job + 1



    # 按时间轴，计算JOB完成数量
    """
    # failed_history = []
    complete_history_time = []
    for t in range(len(edge[0].complete_history)):
        total_complete = 0
        # total_fail = 0
        for k in range(number_of_edge):
            total_complete += edge[k].complete_history[t]
            # total_fail += np.sum(edge[k].failed_history[t])
        complete_history_time.append(total_complete/number_of_edge)
        # failed_history.append(total_fail/number_of_edge)
    """

    # 计算运行结束时，JOB的总完成数量
    total_number_of_complete = 0
    for k in range(number_of_edge):
        total_number_of_complete += edge[k].total_number_of_complete

    partition_history = None
    partition_history = list()

    return cost_total/number_of_job, migration_overhead, total_number_of_complete, \
           np.array(avg_cost_history), complete_history[-1], migration_overhead_history, number_of_job,\
           none_service_history, complete_history_time, migration_overhead_fail
