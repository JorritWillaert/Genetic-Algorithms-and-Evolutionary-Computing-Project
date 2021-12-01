from math import dist
import Reporter
import numpy as np
from typing import List
import random
import matplotlib.pyplot as plt

class Parameters:
	def __init__(self, population_size: int = 100, num_offsprings: int = 100, k: int = 5, its: int = 100):
		self.population_size = population_size
		self.num_offsprings = num_offsprings
		self.k = k
		self.its = its

class Individual:
	def __init__(self, distanceMatrix: np.ndarray, order:List[int]=None, alpha:float=0.05):
		if order is None:
			self.order = np.random.permutation((distanceMatrix.shape)[0])
		else:
			self.order = order
		self.alpha = alpha

def initialization(distanceMatrix: np.ndarray, population_size: int) -> List[Individual]:
	order = np.random.permutation((distanceMatrix.shape)[0])
	test =  [Individual(distanceMatrix, alpha=max(0.01, 0.05+0.02*np.random.randn())) for _ in range(population_size)]
	return test

def fitness(distanceMatrix: np.ndarray, ind: Individual) -> float:
	fit = 0
	for i in range(len(ind.order)):
		elem1 = ind.order[i]
		elem2 = ind.order[(i + 1) % len(ind.order)]
		fit += distanceMatrix[elem1][elem2]
	return fit

def selection(distanceMatrix: np.ndarray, population: List[Individual], k: int) -> Individual:
	"""k-tournament selection"""
	current_min = float('+inf')
	for i in range(k):
		ind = random.choice(population)
		fit = fitness(distanceMatrix, ind)
		if fit < current_min:
			current_min = fit
			best_ind = ind
	return best_ind

def recombination(distanceMatrix: np.ndarray, parent1: Individual, parent2: Individual) -> Individual:
	# Create edge table
	edge_table = [set() for _ in range((distanceMatrix.shape)[0])]
	for i in range((distanceMatrix.shape)[0]):
		edge_table[parent1.order[i]].add(parent1.order[(i + 1) % len(parent1.order)])
		edge_table[parent1.order[i]].add(parent1.order[(i - 1)])
		edge_table[parent2.order[i]].add(parent2.order[(i + 1) % len(parent2.order)])
		edge_table[parent2.order[i]].add(parent2.order[(i - 1)])

	node = random.randint(0, (distanceMatrix.shape)[0] - 1)
	new_order = [node]
	all_nodes = {i for i in range((distanceMatrix.shape)[0])}

	while len(new_order) < (distanceMatrix.shape)[0]:
		for edge_set in edge_table:
			edge_set.discard(node)
		
		if edge_table[node] != set():
			shortest = float('+inf')
			shortest_ones = set()
			for neighbor_node in edge_table[node]:
				if len(edge_table[neighbor_node]) < shortest:
					shortest = len(edge_table[neighbor_node])
					shortest_ones = {neighbor_node}
				elif len(edge_table[neighbor_node]) == shortest:
					shortest_ones.add(neighbor_node)
			chosen_one = random.sample(shortest_ones, 1)[0] # Choose a random element from the shortest ones.
			shortest_ones.remove(chosen_one)

		else:
			possible_ones = all_nodes - set(new_order)
			chosen_one = random.sample(possible_ones, 1)[0] # Choose a random element from the possible ones.
			possible_ones.remove(chosen_one) 

		node = chosen_one
		new_order.append(node)
	beta = 2 * random.random() - 0.5 # Number between -0.5 and 3.5
	alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
	alpha = max(0.01, alpha)
	return Individual(distanceMatrix, order=new_order, alpha=alpha)

def mutation(individual: Individual) -> Individual: # Maybe make this operator in-place
	"""Example mutation: randomly choose 2 indices and swap them."""   
	if random.random() < individual.alpha:
		i = random.randint(0, len(individual.order) - 1)
		j = random.randint(0, len(individual.order) - 1)
		individual.order[i], individual.order[j] = individual.order[j], individual.order[i]
	return individual

def elimination(distanceMatrix: np.ndarray, population: List[Individual], offsprings: List[Individual], lambd: int) -> List[Individual]:
    """Mu + lambda elimination"""
    combined = population + offsprings  
    combined_with_fitness = {}
    for ind in combined:
        combined_with_fitness[ind] = fitness(distanceMatrix, ind)
    sorted_combined = [k for k, _ in sorted(combined_with_fitness.items(), key=lambda x:x[1], reverse=False)]
    return sorted_combined[:lambd]

def swap_edges(ind: Individual, first: int, second: int) -> List[int]:
    """Swap two edges in a circle.
    Image the cycle (A, B, C, ..., Y, Z). If you swap the edges between C-D and Y-Z, 
    then the new cycle becomes (A, B, C, Y , X, W, V, ..., E, D, Z, A)."""  
    return np.concatenate((ind.order[0:first],
                           ind.order[second: - len(ind.order) + first - 1: -1],
                           ind.order[second + 1: len(ind.order)]))

def local_search_operator_2_opt(distanceMatrix: np.ndarray, ind: Individual) -> Individual:
    """Local search operator, which makes use of 2-opt. Swap two edges within a cycle."""
    best_ind = ind
    best_fitness = fitness(distanceMatrix, ind)
    for first in range(1, len(ind.order) - 2):
        for second in range(first + 1, len(ind.order) - 1):
            new_order = swap_edges(ind, first, second)
            new_ind = Individual(distanceMatrix, new_order, ind.alpha)
            new_fitness = fitness(distanceMatrix, new_ind)
            if new_fitness < best_fitness:
                best_ind = new_ind
                best_fitness = new_fitness
    return best_ind

class r0652971:
	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename: str):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		p = Parameters(population_size=100, num_offsprings=100, k=5, its=100)

		population = initialization(distanceMatrix, p.population_size)

		it = 0

		x = range(p.its)

		best_fitnesses = []
		mean_fitnesses = []

		while( it < p.its ): 

			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0

			# timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			# if timeLeft < 0:
			# 	break

			offsprings = []
			for offspring in range(p.num_offsprings):
				parent1 = selection(distanceMatrix, population, p.k)
				parent2 = selection(distanceMatrix, population, p.k)
				offspring = recombination(distanceMatrix, parent1, parent2)
				mut_offspring = mutation(offspring) # Maybe try to mutate list 'in-place' in the future (without return argument)
				ind_after_local_search = local_search_operator_2_opt(distanceMatrix, mut_offspring)
				offsprings.append(ind_after_local_search)
			
			# Mutation of the seed individuals
			for i, seed_individual in enumerate(population):
				mutated_ind = mutation(seed_individual) # Maybe try to mutate list 'in-place' in the future (without return argument)

			population = elimination(distanceMatrix, population, offsprings, p.num_offsprings)

			fitnesses = []
			best_fitness = float('+inf')
			for individual in population:
				fit = fitness(distanceMatrix, individual)
				fitnesses.append(fit)
				if fit < best_fitness:
					best_fitness = fit
					best_individual = individual
			mean_fitness = sum(fitnesses) / len(fitnesses)
			print(f"{it}: Mean fitness: {mean_fitness} \t Best fitness: {min(fitnesses)}")
			best_fitnesses.append(best_fitness)
			mean_fitnesses.append(mean_fitness)
			it += 1


		# Your code here.
		# plt.plot(x, mean_fitnesses, label='Mean fitnesses')
		# plt.plot(x, best_fitnesses, label='Best fitnesses')
		# plt.legend()
		# plt.show()
		# plt.savefig('mean_and_best_fitnesses.png')
		return best_fitness

if __name__ == "__main__":
	problem = r0652971()
	current_best = float('+inf')
	best_fitnesses = []
	# problem.optimize('tour29.csv')
	for i in range(50):
		best_fitness = problem.optimize('tour29.csv')
		best_fitnesses.append(best_fitness)
		if best_fitness < current_best:
			current_best = best_fitness
			print(f"Iteration: {i}, current best: {current_best}")
