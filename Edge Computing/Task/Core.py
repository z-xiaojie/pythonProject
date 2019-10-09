class Core:
    def __init__(self):
        self.job_queue = list()
        self.cur_job = None

    def enqueue(self, job):
        self.job_queue.append(job)
