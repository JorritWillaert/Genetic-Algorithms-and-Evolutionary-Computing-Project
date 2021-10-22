"""
In the binary knapsack problem, you are given a set of objects (o1, o2, ..., on),
along with their values (v1, v2, ..., vn)
and their weights (w1, w2, ..., wn).
The goal is to maximize the total value of the selected objects, 
subject to a weight constraint.
"""
import numpy as np
import random
import sys
from time import sleep

class KnapsackProblem:
    def __init__(self, num_objects):
        self.values = [2 ** (np.random.randn()) for _ in range(num_objects)]
        self.weights = [2 ** (np.random.randn()) for _ in range(num_objects)]
        self.capacity = 0.25 * sum(self.weights) # 25 % of the total weight

    def get_capacity(self):
        return self.capacity

class Parameters:
    def __init__(self, population_size=100, num_offspring=100, num_episodes=100, k_tournament_par=5):
        # Population size, number of offsprings, number of episodes & the k-tournament parameter
        self.population_size = population_size
        self.num_offsprings = num_offspring
        self.num_episodes = num_episodes
        self.k = k_tournament_par

class Individual:
    def __init__(self, knapsackproblem=None, order=None, alpha=0.05):
        if order is None:
            if knapsackproblem is None:
                print("Knapsackproblem may not be 'None' if the order is not specified!")
                sys.exit()
            self.order = (np.random.permutation(len(knapsackproblem.values))).tolist()
        else:
            self.order = order
        self.alpha = alpha # Mutation rate
    
    def get_order(self):
        return self.order

def fitness(knapsackproblem, individual):
    individual_order = individual.get_order()
    remaining_capacity = knapsackproblem.get_capacity()
    current_value = 0
    for i in individual_order:
        if knapsackproblem.weights[i] <= remaining_capacity:
            remaining_capacity -= knapsackproblem.weights[i]
            current_value += knapsackproblem.values[i]
        # It is suggested not to break if the current item does not fit in the knapsack.s
        # Forthcoming fitting items are hence still allowed.
    return current_value

def in_knapsack(knapsackproblem, individual):
    individual_order = individual.get_order()
    remaining_capacity = knapsackproblem.get_capacity()
    objects_in_knapsack = set()
    for i in individual_order:
        if knapsackproblem.weights[i] <= remaining_capacity:
            remaining_capacity -= knapsackproblem.weights[i]
            objects_in_knapsack.add(i)
        # It is suggested not to break if the current item does not fit in the knapsack.s
        # Forthcoming fitting items are hence still allowed.
    return objects_in_knapsack

def initialization(kp, population_size):
    return [Individual(knapsackproblem=kp) for _ in range(population_size)]

def mutation(individual):
    """Example mutation: randomly choose 2 indices and swap them."""   
    if random.random() < individual.alpha:
        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        tmp = individual.order[i]
        individual.order[i] = individual.order[j]
        individual.order[j] = tmp
    return individual

def recombination(knapsackproblem, parent1, parent2):
    """Use recombination for sets instead of permutation, since the order determines the elements in the knapsack."""
    s1 = in_knapsack(knapsackproblem, parent1)
    s2 = in_knapsack(knapsackproblem, parent2)

    # Copy intersection to offspring
    offspring = s1.intersection(s2)
    for i in s1.symmetric_difference(s2):
        # Copy the elements in the symmetric offspring with 50 % probability
        if random.random() < 0.5:
            offspring.add(i)
    
    # It doens't matter that the offspring might not fit in the knapsack.
    # Fitness function takes care of this.

    # The elements that are not yet in the offspring, are collected in remainder.
    remaining = set(range(len(knapsackproblem.values))).difference(offspring)
    
    order = []
    for off in offspring:
        order.append(off) # These elements will most likely fit in knapsack, since in front of 'order'.
    for rem in remaining:
        order.append(rem) # However, some of the first of these may be selected as well, if the preceding ones don't fit.
    
    # Make sure the elements from the offspring appear in a random order
    order[0: len(offspring)] = (np.random.permutation(order[0: len(offspring)])).tolist()
    # The following elements (not from the offspring) should also be in random order
    order[len(offspring) : ] = (np.random.permutation(order[len(offspring) : len(knapsackproblem.values)])).tolist()
    
    # Way to assign a new alpha to our child
    beta = 2 * random.random() - 0.5 # Number between -0.5 and 3.5
    alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
    return Individual(order=order, alpha=alpha)

def selection(knapsackproblem, population, k):
    """k-tournament selection"""
    selected = []
    for i in range(k):
        selected.append(random.choice(population))
    current_max = float('-inf')
    for ind in selected:
        if fitness(knapsackproblem, ind) > current_max:
            current_max = fitness(knapsackproblem, ind)
            best_ind = ind
    return best_ind

def elimination(knapsackproblem, population, offsprings, lambd):
    """Mu + lambda elimination"""
    combined = population + offsprings  
    combined_with_fitness = {}
    for ind in combined:
        combined_with_fitness[ind] = fitness(knapsackproblem, ind)
    sorted_combined = [k for k, _ in sorted(combined_with_fitness.items(), key=lambda x:x[1], reverse=True)]
    return sorted_combined[:lambd]

def evolutionary_algorithm(kp: KnapsackProblem, p: Parameters):
    population = initialization(kp, p.population_size)
    fitnesses = []
    for individual in population:
        fitnesses.append(fitness(kp, individual))
    print(f"0: Mean fitness: {sum(fitnesses) / len(fitnesses)} \t Best fitness: {max(fitnesses)}")
    
    for episode in range(p.num_episodes):
        offsprings = []
        for offspring in range(p.num_offsprings):
            parent1 = selection(kp, population, p.k)
            parent2 = selection(kp, population, p.k)
            offspring = recombination(kp, parent1, parent2)
            mut_offspring = mutation(offspring) # Maybe try to mutate list 'in-place' in the future (without return argument)
            offsprings.append(mut_offspring)
        
        # Mutation of the seed individuals
        for i, seed_individual in enumerate(population):
            population[i] = mutation(seed_individual) # Maybe try to mutate list 'in-place' in the future (without return argument)
        
        population = elimination(kp, population, offsprings, p.num_offsprings)

        fitnesses = []
        for individual in population:
            fitnesses.append(fitness(kp, individual))
        print(f"{episode}: Mean fitness: {sum(fitnesses) / len(fitnesses)} \t Best fitness: {max(fitnesses)}")

def tests():
    kp = KnapsackProblem(25)
    ind = Individual(knapsackproblem=kp)
    print(f"Knapsack values: {kp.values}")
    print(f"Knapsack weights = {kp.weights}")
    print(f"Knapsack capacity = {kp.capacity}")
    print(f"Individual: {ind.order}")
    print(f"Objective value of the individual: {fitness(kp, ind)}")
    print("\nTest initialization")

    population = initialization(kp, 3)
    for ind in population:
        print(f"Individual: {ind.order}")
        print(f"Objective value of the individual: {fitness(kp, ind)}")
        print(f"The objects in the knapsack: {in_knapsack(kp, ind)}")
    
    for i in range(20):
        population[0] = mutation(population[0])
        print(population[0].order)

    print("\nTest recombination")

    offspring = recombination(kp, population[0], population[1])
    print(f"Order of parent 1: {population[0].order}")
    print(f"The objects in the knapsack of parent 1: {in_knapsack(kp, population[0])}")
    print(f"Order of parent 2: {population[1].order}")
    print(f"The objects in the knapsack of parent 2: {in_knapsack(kp, population[1])}")
    print(f"Order of the offspring: {offspring.order}")
    print(f"The objects in the knapsack of the offspring: {in_knapsack(kp, offspring)}")
    print(f"Offspring alpha: {offspring.alpha}")

    print("\nTrest selection")

    population = initialization(kp, 25)
    print(selection(kp, population, 5).order)

    print("\nTest elimination")
    simulated_parents = initialization(kp, 3)
    simulated_children = initialization(kp, 3)
    outcome = elimination(kp, simulated_parents, simulated_children, 3)
    for ind in outcome:
        print(ind.order)

def main():
    p = Parameters(population_size=100, num_offspring=100, num_episodes=25, k_tournament_par=5)
    kp = KnapsackProblem(50)
    
    print(f"Knapsack values: {kp.values}")
    print(f"Knapsack weights = {kp.weights}")
    print(f"Knapsack capacity = {kp.capacity}")

    evolutionary_algorithm(kp, p)


if __name__ == '__main__':
    # tests()
    main()