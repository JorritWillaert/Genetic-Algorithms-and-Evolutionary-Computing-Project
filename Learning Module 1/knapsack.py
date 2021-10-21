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

from numpy.core.fromnumeric import cumprod

class KnapsackProblem:
    def __init__(self, num_objects):
        self.values = [2 ** (np.random.randn()) for _ in range(num_objects)]
        self.weights = [2 ** (np.random.randn()) for _ in range(num_objects)]
        self.capacity = 0.25 * sum(self.weights) # 25 % of the total weight

    def get_capacity(self):
        return self.capacity

class Individual:
    def __init__(self, knapsackproblem=None, order=None, alpha=0.05):
        if order is None:
            if knapsackproblem is None:
                print("Knapsackproblem may not be 'None' if the order is not specified!")
                sys.exit()
            self.order = np.random.permutation(len(knapsackproblem.values))
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
        i = random.randint(0, len(individual.order))
        j = random.randint(0, len(individual.order))
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
    order[0: len(offspring)] = np.random.permutation(order[0: len(offspring)])
    # The following elements (not from the offspring) should also be in random order
    order[len(offspring) : ] = np.random.permutation(order[len(offspring) : len(knapsackproblem.values)])
    
    # Way to assign a new alpha to our child
    beta = 2 * random.random() - 0.5 # Number between -0.5 and 3.5
    alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
    return Individual(order=order, alpha=alpha)

def selection(knapsackproblem, population):
    """k-tournament selection"""
    k = 5
    selected = []
    for i in range(k):
        selected.append(random.choice(population))
    current_max = float('-inf')
    for ind in selected:
        if fitness(knapsackproblem, ind) > current_max:
            current_max = fitness(knapsackproblem, ind)
            best_ind = ind
    return best_ind

def evolutionary_algorithm(kp):
    # Population size, number of offsprings
    population_size = 100
    num_offsprings = 100
    num_episodes = 100

    population = initialization(kp, population_size)
    for episode in range(num_episodes):
        offsprings = []
        for offspring in range(num_offsprings):
            parent1 = selection(kp, population)
            parent2 = selection(kp, population)
            offspring = recombination(kp, parent1, parent2)
            mut_offspring = mutation(offspring) # Maybe try to mutate list 'in-place' in the future (without return argument)
            offsprings.append(mut_offspring)
        
        # Mutation of the seed individuals
        for i, seed_individual in enumerate(population):
            population[i] = mutation(seed_individual) # Maybe try to mutate list 'in-place' in the future (without return argument)

        population = elimination(kp, population, offsprings)

        fitnesses = []
        for individual in population:
            fitnesses.append(fitness(kp, individual))
        print(f"Mean fitness: {sum(fitnesses) / len(fitnesses)}")
        print(f"Best fitness: {max(fitnesses)}")
    

def tests():
    kp = KnapsackProblem(10)
    ind = Individual(knapsackproblem=kp)
    print(f"Knapsack values: {kp.values}")
    print(f"Knapsack weights = {kp.weights}")
    print(f"Knapsack capacity = {kp.capacity}")
    print(f"Individual: {ind.order}")
    print(f"Objective value of the individual: {fitness(kp, ind)}")
    print()

    population = initialization(kp, 3)
    for ind in population:
        print(f"Individual: {ind.order}")
        print(f"Objective value of the individual: {fitness(kp, ind)}")
        print(f"The objects in the knapsack: {in_knapsack(kp, ind)}")
    
    for i in range(20):
        population[0] = mutation(population[0])
        print(population[0].order)

    print()

    offspring = recombination(kp, population[0], population[1])
    print(f"Order of parent 1: {population[0].order}")
    print(f"The objects in the knapsack of parent 1: {in_knapsack(kp, population[0])}")
    print(f"Order of parent 2: {population[1].order}")
    print(f"The objects in the knapsack of parent 2: {in_knapsack(kp, population[1])}")
    print(f"Order of the offspring: {offspring.order}")
    print(f"The objects in the knapsack of the offspring: {in_knapsack(kp, offspring)}")
    print(f"Offspring alpha: {offspring.alpha}")

    print()

    population = initialization(kp, 25)
    print(selection(kp, population).order)


if __name__ == '__main__':
    tests()