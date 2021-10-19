"""
In the binary knapsack problem, you are given a set of objects (o1, o2, ..., on),
along with their values (v1, v2, ..., vn)
and their weights (w1, w2, ..., wn).
The goal is to maximize the total value of the selected objects, 
subject to a weight constraint.
"""
import numpy as np

class KnapsackProblem:
    def __init__(self, num_objects):
        self.values = [2 ** (np.random.randn()) for _ in range(num_objects)]
        self.weights = [2 ** (np.random.randn()) for _ in range(num_objects)]
        self.capacity = 0.25 * sum(self.weights) # 25 % of the total weight

    def get_capacity(self):
        return self.capacity

class Individual:
    def __init__(self, knapsackproblem):
        self.order = np.random.permutation(len(knapsackproblem.values))
    
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
    #TODO 40:28 --> Also use set

    individual_order = individual.get_order()
    remaining_capacity = knapsackproblem.get_capacity()
    current_value = 0
    objects_in_knapsack = []
    for i in individual_order:
        if knapsackproblem.weights[i] <= remaining_capacity:
            remaining_capacity -= knapsackproblem.weights[i]
            current_value += knapsackproblem.values[i]
            objects_in_knapsack.append(i)
        # It is suggested not to break if the current item does not fit in the knapsack.s
        # Forthcoming fitting items are hence still allowed.
    return current_value, objects_in_knapsack

def initialization(kp, population_size):
    return [Individual(kp) for _ in range(population_size)]

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
            offspring = recombination(parent1, parent2)
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
    ind = Individual(kp)
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


if __name__ == '__main__':
    tests()