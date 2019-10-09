
class Job:
    'Job'
    def __init__(self, release_time, exe_time, job_size, interval, deadline, dead_time, id, serial_number, cloud_id, core_id):

        # params
        self.release_time = release_time
        self.exe_time = exe_time
        self.worst_cast_time = exe_time
        self.job_size = job_size
        self.dead_time = dead_time
        self.deadline = deadline
        self.interval = interval
        self.id = id
        self.serial_number = serial_number

        # job scheduling
        self.cloud_id = cloud_id
        self.cloud_id = core_id

        # data transmission between user and edge cloud
        self.transmitted_data = 0
        self.finish_time = -1
        self.state = "Released"

        self.summary(release_time)

    # all data moved to edge core, ready to execute
    def ready_to_exe(self):
        if self.transmitted_data >= self.job_size:
            return True
        else:
            return False

    def exe(self, cur_time):
        # job status counter
        self.exe_time = self.exe_time - 1
        if self.dead_time <= cur_time and self.exe_time > 0:
            self.state = "Failed"
        else:
            if self.exe_time <= 0:
                self.state = "Complete"
                self.finish_time = cur_time
            else:
                self.state = "Computing"
        self.summary(cur_time)
        return self.state

    def transmit(self, rate, cur_time):
        # 5 MB  <-  4.5 MB
        # 2 Mbps
        # 6.5 - 5 = 1.5
        # 1.5 / 2 = overflow
        if self.transmitted_data < self.job_size:
            self.transmitted_data += rate
        if self.ready_to_exe():
            self.state = "Transmitted"
        else:
            self.state = "Transmitting"
        self.summary(cur_time)
        return self.state, (self.transmitted_data - self.job_size)/rate

    def summary(self, cur_time):
        label = ""
        if self.state == "Transmitting":
            label = ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        if self.state == "Transmitted":
            label = "............................"
        if self.state == "Computing":
            label = "----------------------------"
        if self.state == "Complete" or self.state == "Failed":
            label = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        if self.state == "Released":
            label = "++++++++++++++++++++++++++++"

        #self.info(cur_time, label)

    def info(self, cur_time, label):
        print("Time(", cur_time, ")", label, "User", (self.id, self.release_time, self.dead_time, self.interval),
              ", Job(", self.serial_number, "), REQ",
              (self.worst_cast_time, self.exe_time),   "Data", (self.job_size, self.transmitted_data), self.state)



