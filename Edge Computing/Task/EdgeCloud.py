from Task.Node import Node
from Task.Check import condition_one, condition_two
from Task.Core import Core


class EdgeCloud(Node):
    def __init__(self, cur_time, users, number_of_core=3, bandwidth_max=1):
        super(EdgeCloud, self).__init__(cur_time)
        self.number_of_core = number_of_core
        self.bandwidth_max = bandwidth_max
        self.bandwidth_real = bandwidth_max

        # compute resource
        self.cores = list()
        for i in range(self.number_of_core):
            self.cores.append(Core())

        # service
        self.users = users
        self.user_to_core = []
        self.capacity = number_of_core

        # test feasible
        # partition = [0,0,0,1,1,0]  min_rate = 0.4
        self.history = list()

        # summary
        self.failed_history = []
        self.complete_history = []
        self.release_history = []

        self.number_of_job = 0
        self.number_of_complete = 0
        self.number_of_failed = 0

        self.total_number_of_job = 0
        self.total_number_of_complete = 0
        self.total_number_of_failed = 0

    ########################
    #  network information
    ########################
    def get_number_of_users(self):
        if self.users is None:
            return 0
        return len(self.users)

    def allocate_users(self, user_partition, users):
        self.users = []
        for n in range(len(user_partition)):
            if user_partition[n] == self.id:
                self.users.append(users[n])

    def allocate_fair_rate(self, adjust):
        if self.get_number_of_users() == 0:
            rate = 0
        else:
            rate = self.bandwidth_max*adjust / self.get_number_of_users()
        for n in range(self.get_number_of_users()):
            self.users[n].set_allocated_rate(rate)

    ########################
    #  job exe
    ########################
    def receive_job(self, job, core_id):
        self.cores[core_id].enqueue(job)
        self.number_of_job = self.number_of_job + 1
        self.total_number_of_job = self.total_number_of_job + 1

    def exe_job(self):
        for i in range(self.number_of_core):
            self.cores[i].exe()

    ########################
    #  job scheduling test
    #  user_to_core = { "user_id" ： n, "core_id" : m }
    ########################
    def job_scheduling(self, adjust):
        if adjust <= 0:
            return False
        new_user_to_core = []
        self.allocate_fair_rate(adjust)
        self.users.sort(key=lambda x: x.get_deadline_with_rate())
        can_scheduling = 0
        for n in range(self.get_number_of_users()):
            for k in range(self.number_of_core):
                assigned = []
                for item in new_user_to_core:
                    if item["core_id"] == k:
                        assigned.append(self.find_user_by_id(item["user_id"]))
                #
                if condition_one(self.users[n], assigned) and condition_two(self.users[n], assigned):
                    # assign user to cloud
                    new_user_to_core.append({"user_id": self.users[n].id, "core_id": k})
                    can_scheduling += 1
                    break
            """
            if not can_scheduling:
                return False
            """
        self.user_to_core = new_user_to_core
        return can_scheduling == len(self.users)

    def find_user_by_id(self, user_id):
        for n in range(self.get_number_of_users()):
            if self.users[n].id == user_id:
                return self.users[n]

    def find_core_to_user_by_id(self, user_id):
        for item in self.user_to_core:
            if item["user_id"] == user_id:
                return item["core_id"]
        return None

    ########################
    #  timer
    ########################
    def step(self, cur_time):
        self.cur_time = cur_time

    def exe(self):
        for m in range(self.number_of_core):
            core = self.cores[m]
            if len(core.job_queue) == 0:  # and self.cur_job is None:
                continue
            core.job_queue.sort(key=lambda x: x.dead_time)
            core.cur_job = core.job_queue[0]
            if core.cur_job is not None:
                state = core.cur_job.exe(self.cur_time)
                if state == "Complete" or state == "Failed":
                    core.job_queue.remove(core.cur_job)
                    core.cur_job = None
                    if state == "Complete":
                        self.number_of_complete = self.number_of_complete + 1
                        self.total_number_of_complete = self.total_number_of_complete + 1
                    if state == "Failed":
                        self.number_of_failed = self.number_of_failed + 1
                        self.total_number_of_failed = self.total_number_of_failed + 1

    def clear_data(self):
        self.number_of_job = 0
        self.number_of_complete = 0
        self.number_of_failed = 0
