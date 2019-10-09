import random
from Task.Chromosome import Chromosome


# Class to represent biological processes
class BiologicalProcessManager:
		'''
			Crossover Function

			- The process of One-Point crossover is exercised in this function.
		'''
		def crossover(crossover_rate, parentOne, parentTwo):
			random_probability = random.uniform(0, 1)
			if random_probability > crossover_rate:
				pivot = random.randint(0, len(parentOne.genotype_representation)-1)
				child_one_genotype = parentOne.genotype_representation[:pivot] + parentTwo.genotype_representation[pivot:]
				child_two_genotype = parentTwo.genotype_representation[:pivot] + parentOne.genotype_representation[pivot:]
				child_one = Chromosome(parentOne.alpha, parentOne.number_of_edge, parentOne.number_of_user, parentOne.pre_tf, child_one_genotype)
				child_two = Chromosome(parentOne.alpha, parentOne.number_of_edge, parentOne.number_of_user, parentOne.pre_tf, child_two_genotype)
				return child_one, child_two
			else:
				return parentOne, parentTwo


		'''
			Mutation function

			- The process of Random Resetting is exercised in this function.
		'''
		def mutate(mutation_rate, child, candidates):
			for index, position in enumerate(child.genotype_representation):
				random_probability = random.uniform(0, 1)
				'''
					(Random Resetting) "Flip" the position with another knapsack if probability < mutation_rate
				'''
				if random_probability > mutation_rate:
					child.genotype_representation[index] = candidates[index][random.randint(0, len(candidates)-1)]





