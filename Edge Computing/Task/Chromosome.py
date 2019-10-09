import random
import numpy as np
from Task import Optimization
import copy


# Class to represent chromosome
class Chromosome:
    def __init__(self, alpha, number_of_edge, number_of_user, pre_tf, genotype_representation=None):
        self.fitness = None  # Chromosomes fitness
        self.number_of_edge = number_of_edge
        self.number_of_user = number_of_user
        self.alpha = alpha
        self.pre_tf = pre_tf
        self.tf = 0
        self.rf = 0
        if genotype_representation is None:
            self.genotype_representation = [random.randint(0, number_of_edge - 1) for x in range(0, number_of_user)]
        else:
            self.genotype_representation = copy.deepcopy(genotype_representation)
        self.length_of_encoding = len(self.genotype_representation)

    """
    Generates a fitness for all the chromosomes by aggregating their benefits/values
    """
    def generate_fitness(self, edges, users, pre_partition):
        """
        Make a copy of the knapsack list to be used to evaluate if objects in the chromsome
                    exceed C using the 'amountUsed' attribute
        """
        # print("ORIGINAL CHROM: {}".format(self.genotype_representation))
        #start = time.time()
        #print("time:", time.time() - start)
        fitness_score = 0
        """
        for i, placement_of_object in enumerate(self.genotype_representation):
                # print(self.genotype_representation)
                for edge in edges:
                    if edge.id == placement_of_object:
                        # if it's over the capacity, change it's bag and revaluate
                        edge.allocate_users(self.genotype_representation, users)
                        adjust = edge.avg_data[int(edge.cur_time / edge.interval)]
                        if not edge.job_scheduling(adjust):
                            tries = 0
                            while True:
                                if tries > self.number_of_edge * 50:
                                    break
                                else:
                                    tries += 1
                                self.genotype_representation[i] = random.randint(0, self.number_of_edge - 1)
                                current_edge = next(
                                    (sack for sack in edges if sack.id == self.genotype_representation[i]),
                                    None)
                                current_edge.allocate_users(self.genotype_representation, users)
                                adjust = current_edge.avg_data[int(current_edge.cur_time / current_edge.interval)]
                                if not current_edge.job_scheduling(adjust):
                                    continue
                                else:
                                    break
        """
        # update the chromosomes fitness
        self.fitness = self.get_fitness(edges, users, pre_partition)

    def get_density(self, edge, user):
        number_of_assigned = np.count_nonzero(np.array(self.genotype_representation) == edge.id)
        rate = edge.avg_data[int(edge.cur_time / edge.interval)] * edge.bandwidth_max / number_of_assigned
        user.set_allocated_rate(rate)
        return user.get_density_with_rate()

    def get_fitness(self, edge, users, pre_partition):
        self.rf,  self.tf, self.fitness = Optimization.get_fitness(self.alpha, edge, users, self.pre_tf,
                                                                   self.genotype_representation, pre_partition, self.number_of_user)
        return self.fitness


