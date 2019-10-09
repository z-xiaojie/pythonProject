from Task.Chromosome import Chromosome
from Task import Optimization


class Assignment:
    def __init__(self, uer_partition, bandwidth, tf):
        self.uer_partition = uer_partition
        self.bandwidth = bandwidth
        self.tf = tf
        self.used = 0

    """
           old = [0.8 0.7 0.5]
           new = [0.7 0.4 0.3]
           diff = 0.1 + （-0.3)  + (-0.2)  = -0.4

           old = [0.8 0.7 0.5]
           new = [0.5 0.4 1]
           diff = (-0.3) + （-0.3)  + 1  = 0.4
       """
    def compute_difference(self, bandwidth, min_diff):
        d = 0
        better = 0
        for i in range(len(bandwidth)):
            d += bandwidth[i] - self.bandwidth[i]
            if abs(bandwidth[i] - self.bandwidth[i]) <= min_diff:
                better = better + 1
        return abs(d), better


class AssignmentMemory:
    def __init__(self, max_size=50, min_diff=0.1, min_better=None):
        self.past_assignment = list()
        self.max_size = max_size
        self.min_diff = min_diff
        self.min_better = min_better

    def search_past_assignment(self, alpha, pre_tf, edge, users, number_of_edge, number_of_user, pre_partition,
                               bandwidth, pre_feasible):
        if pre_feasible:
            max_fitness = 0
        else:
            max_fitness = -9999
        valid_past = None
        # number of job in one prediction window
        number_of_job = 0
        for u in users:
            number_of_job += edge[0].interval / u.interval
        # search past assignment
        self.past_assignment.sort(key=lambda x: x.tf)
        for past in self.past_assignment:
            diff, better = past.compute_difference(bandwidth, self.min_diff)
            if better >= self.min_better:
                rf = Optimization.migration_overhead(past.uer_partition, pre_partition, number_of_user)
                tf = Optimization.estimate_data_transfer(edge, users, number_of_user, past.uer_partition)
                fitness = number_of_job * alpha * (pre_tf - tf)/number_of_user - (1 - alpha) * rf
                if fitness > max_fitness and Optimization.feasible(edge, users, number_of_user, number_of_edge, past.uer_partition):
                    valid_past = past
                    max_fitness = fitness
        if valid_past is not None:
            valid_past.used += 1
            return valid_past.uer_partition, max_fitness, valid_past.bandwidth
        else:
            return None, 0, None

    def check_exist(self, uer_partition, bandwidth, fitness):
        for past in self.past_assignment:
            same = True
            for i in range(len(past.uer_partition)):
                if past.uer_partition[i] != uer_partition[i]:
                    same = False
            if same:
                past.uer_partition = uer_partition
                past.bandwidth = bandwidth
                past.fitness = fitness
                return True
        return False

    def replace(self, uer_partition, bandwidth, tf):
        # 取代最少使用的
        assignment = Assignment(uer_partition, bandwidth, tf)
        exist = False
        for past in self.past_assignment:
            diff, better = past.compute_difference(bandwidth, 0.005)
            if better >= self.min_better:
                past.used += 1
                exist = True
                break
        if exist:
            #print("existing past_assignment", uer_partition, bandwidth, tf)
            x = 1
        else:
            if len(self.past_assignment) < self.max_size:
                self.past_assignment.append(assignment)
            else:
                self.past_assignment.sort(key=lambda x: x.used)
                del self.past_assignment[0]
                self.past_assignment.append(assignment)
            #print("insert into past_assignment", uer_partition, bandwidth, tf)

    def initial_population(self, alpha, policy, ljbj_assignment, pre_tf, size, edge, users, number_of_edge, number_of_user, pre_partition):
        population = []
        j = 0
        self.past_assignment.sort(key=lambda x: x.tf, reverse=True)
        while len(population) < size:
            if len(population) == 0 and policy == 4 and pre_partition is not None:
                new_chromosome = Chromosome(alpha, number_of_edge, number_of_user, pre_tf,
                                            genotype_representation=list(pre_partition))
            elif len(population) <= int(size/2) and policy == 4:
                new_chromosome = Chromosome(alpha, number_of_edge, number_of_user, pre_tf,
                                            genotype_representation=list(ljbj_assignment))
            elif j < len(self.past_assignment) and policy == 4:
                # Create a new chromosome
                new_chromosome = Chromosome(alpha, number_of_edge, number_of_user, pre_tf,
                                            genotype_representation=list(self.past_assignment[j].uer_partition))
                j = j + 1
            else:
                new_chromosome = Chromosome(alpha, number_of_edge, number_of_user, pre_tf)
            new_chromosome.generate_fitness(edge, users, pre_partition)
            population.append(new_chromosome)
        return population


        """
           elif j < len(self.past_assignment):
               # Create a new chromosome
               new_chromosome = Chromosome(alpha, number_of_edge, number_of_user, pre_tf,
                                           genotype_representation=list(self.past_assignment[j].uer_partition))
               j = j + 1
           """
