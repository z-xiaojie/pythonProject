from Task.Chromosome import Chromosome
from collections import namedtuple
import copy


class Population:

    def __init__(self, size):
        self.populationSize = size

        self.number_of_edge = 0
        self.number_of_user = 0
        self.edge = []
        self.users =[]

        self.Phenotype = namedtuple('Phenotype', 'id exe_time job_size interval deadline')
        self.knapsackList = []  # list of knapsacks
        self.knapSackEvaluationList = []  # used for generating fitness of chromosomes
        self.population = []

    def select_parents(self, tournament):
        """
         Tournament selection is being used to find two parents
        """
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        first_fittest_indiv = copy.deepcopy(tournament[0])
        second_fittest_indiv = copy.deepcopy(tournament[1])
        # print("FIRST: {},  SECOND: {}".format(first_fittest_indiv.fitness,second_fittest_indiv.fitness))
        return first_fittest_indiv, second_fittest_indiv

    def initialize_population(self, alpha, pre_tf, edge, users, number_of_edge, number_of_user, pre_partition):
        self.number_of_edge = number_of_edge
        self.number_of_user = number_of_user
        self.edge = edge
        self.users = users

        # Create the initial population
        #start = time.time()
        for j in range(0, self.populationSize):
            # Create a new chromosome
            new_chromosome = Chromosome(alpha, self.number_of_edge, self.number_of_user, pre_tf)
            # Evaluate each chromosome
            #print(new_chromosome.genotype_representation)
            new_chromosome.generate_fitness(self.edge, self.users, pre_partition)
            # Add the chromsome to the population
            self.population.append(new_chromosome)
        #print("time:", time.time() - start)

    def replace(self, child, edge, users, pre_partition):
        fitness = child.get_fitness(edge, users, pre_partition)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if fitness > self.population[-1].fitness:
            del self.population[-1]
            self.population.append(child)


