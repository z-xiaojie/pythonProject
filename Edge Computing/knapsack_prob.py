from Population import Population
from Chromosome import  Chromosome
import Optimization
from genetic_toolkit import BiologicalProcessManager
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import Optimization
import copy


def find_the_best(pre_feasible, pre_partition, population, edge, users, number_of_user, number_of_edge):
	population.sort(key=lambda x: x.fitness, reverse=True)
	# if previous partition is feasible, do  migrate only when new assignment has a positive fitness and is also feasible
	for i in range(len(population)):
		feasible = Optimization.feasible(edge, users, number_of_user, number_of_edge,
																population[i].genotype_representation)
		if pre_feasible:
			if population[i].fitness > 0 and feasible:
				return feasible, population[i]
		else:
			if feasible:
				return feasible, population[i]
	return False, population[0]


def find_the_worst(population):
	min_fitness = -999
	min_item = None
	index = -1
	for i in range(len(population)):
		if population[i].fitness < min_fitness:
			min_fitness = population[i].fitness
			min_item = population[i]
			index = i
	return index, min_item


def opt(alpha, policy, ljbj_assignment, pre_tf, pre_feasible,  edge, users, number_of_edge, number_of_user, pre_partition, memory, bandwidth):

	# Global Variables
	crossover_rate, population_size = 0.1, number_of_user*3
	# Initialize population with random candidate solutions
	pop = Population(population_size)
	pop.number_of_edge = number_of_edge
	pop.number_of_user = number_of_user
	pop.edge = edge
	pop.users = users
	pop.population = memory.initial_population(alpha, policy, ljbj_assignment, pre_tf, population_size, edge, users, number_of_edge,
											   number_of_user, pre_partition)

	# Set the mutation rate
	mutation_rate = 0.3
	# Get a reference to the number of knapsacks
	numberOfKnapsacks = pop.number_of_edge
	generation_counter = 0
	while generation_counter <= number_of_edge*2:
		#current_population_fitnesses = np.array([chromosome.fitness for chromosome in pop.population])
		new_generation = []
		# print("CURRENT GEN FITNESS:", current_population_fitnesses)
		#print(generation_counter, "CURRENT GEN FITNESS:", np.max(current_population_fitnesses))
		# maxxxx = np.max(current_population_fitnesses)
		while len(new_generation) < population_size:
			# Create tournament for tournament selection process
			# pop.population.sort(key=lambda x: x.tf)
			tournament = [pop.population[random.randint(1, pop.populationSize - 1)] for individual in
						  range(1, pop.populationSize)]

			# pop.population.sort(key=lambda x: x.fitness, reverse=True)

			# Obtain two parents from the process of tournament selection
			parent_one, parent_two = pop.select_parents(
				tournament)  # copy.deepcopy(pop.population[0]), copy.deepcopy(pop.population[1])

			# Create the offspring from those two parents
			child_one, child_two = BiologicalProcessManager.crossover(crossover_rate, parent_one, parent_two)
			# print(parent_one.genotype_representation)
			# print(child_one.genotype_representation)

			# Try to mutate the children
			BiologicalProcessManager.mutate(mutation_rate, child_one, numberOfKnapsacks)
			BiologicalProcessManager.mutate(mutation_rate, child_two, numberOfKnapsacks)
			# index3, best3 = find_the_best(pop.population)
			# print("3 New GEN FITNESS:", best3.fitness, best3.genotype_representation, index3)

			# Evaluate each of the children
			child_one.generate_fitness(edge, users, pre_partition)
			child_two.generate_fitness(edge, users, pre_partition)

			# index4, best4 = find_the_best(pop.population)
			# print("4 New GEN FITNESS:", best4.fitness, best4.genotype_representation, index4)

			# Add the children to the new generation of chromsomes
			# if random.uniform(0, 1) < 0.5 or Optimization.feasible(edge, users, number_of_user, number_of_edge, child_one.genotype_representation):
			# if random.uniform(0, 1) < 0.5 or Optimization.feasible(edge, users, number_of_user, number_of_edge, child_two.genotype_representation):

			# print(child_one.fitness, child_two.fitness)
			# if child_one.fitness > parent_one.fitness:
			new_generation.append(child_one)
			# if child_two.fitness > parent_two.fitness:
			new_generation.append(child_two)

		generation_counter += 1
		for item in new_generation:
			pop.population.append(item)
		pop.population.sort(key=lambda x: x.fitness, reverse=True)
		pop.population = pop.population[:population_size]

	# update solution memory with lowest transmission cost
	pop.population.sort(key=lambda x: x.tf)
	memory.replace(pop.population[0].genotype_representation, bandwidth, pop.population[0].tf)

	feasible, best = find_the_best(pre_feasible, pre_partition, pop.population, edge, users, number_of_user,
								   number_of_edge)
	if best is not None:
		# print(feasible, best.fitness, best.tf)
		return 0, np.array(best.genotype_representation), best.fitness, best.tf, best.rf
	else:
		return 0, None, -1, -1, -1
