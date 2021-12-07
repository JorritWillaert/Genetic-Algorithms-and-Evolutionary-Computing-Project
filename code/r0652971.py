from math import dist
from operator import length_hint, ne, pos
from os import wait
from time import sleep
from numba import jit

from numpy.core.numeric import _ones_like_dispatcher, ones_like
import Reporter
import numpy as np
from typing import List, Tuple
import random
import matplotlib.pyplot as plt
import cProfile

# TODO:
# Probably beneficial to always start from the first city
# Lists to Numpy arrays

class Parameters:
    def __init__(self, population_size: int = 100, num_offsprings: int = 100, k: int = 5):
        self.population_size = population_size
        self.num_offsprings = num_offsprings
        self.k = k

class Individual:
    def __init__(self, distanceMatrix: np.ndarray, order: List[int]=None, alpha: float=0.05):
        if order is None:
            self.order = np.random.permutation((distanceMatrix.shape)[0])
        else:
            self.order = order
        self.alpha = alpha

def initialization(distanceMatrix: np.ndarray, population_size: int) -> List[Individual]:
    individuals = [None] * population_size
    percentage_greedily = 0.80 # TODO: In Parameters class
    greedily_number = int(population_size * percentage_greedily)
    for i in range(greedily_number):
        individuals[i] = greedily_initialize_individual(distanceMatrix)
    for i in range(greedily_number, population_size):
        individuals[i] = Individual(distanceMatrix, alpha=max(0.01, 0.05+0.02*np.random.randn()))
    print("Initialization ended")
    return individuals

def greedily_initialize_individual(distanceMatrix: np.ndarray) -> Individual:
    length = (distanceMatrix.shape)[0]	
    
    i = 0
    while i != length:
        order = np.negative(np.ones((length), dtype=np.int))
        city = np.random.randint(0, length - 1)
        order[0] = city
        i = 1
        while i < length:
            possibilities = set(range(length)) - set([elem for elem in order if elem >= 0])
            min_distance = float("+inf")
            for pos in possibilities:
                distance = distanceMatrix[city][pos]
                if distance < min_distance:
                    min_distance = distance
                    new_city = pos
            if min_distance == float("+inf"):
                break
            city = new_city
            order[i] = city
            i += 1
    return Individual(distanceMatrix, order=order, alpha=max(0.01, 0.05+0.02*np.random.randn()))

def partial_fitness_one_value(distanceMatrix: np.ndarray, frm: int, to: int):
    distance = distanceMatrix[frm][to]
    if distance != float("inf"):
        return distance
    return 10_000_000.0


def partial_fitness_without_looping_back(distanceMatrix: np.ndarray, partial_order: np.ndarray) -> float:
    if (len(partial_order) == 0):
        return 0.0
    fit = 0.0
    num_of_infinities = 0
    for i in range(len(partial_order) - 1):
        elem1 = partial_order[i]
        elem2 = partial_order[i + 1]
        if fit == float("+inf"):
            num_of_infinities += 1
        else:
            fit += distanceMatrix[elem1][elem2]
    return fit, num_of_infinities

def fitness(distanceMatrix: np.ndarray, order: np.ndarray) -> float:
    fit = 0.0
    for i in range(len(order)):
        elem1 = order[i]
        elem2 = order[(i + 1) % len(order)]
        fit += distanceMatrix[elem1][elem2]
        if fit == float("+inf"):
            return fit
    return fit

def selection(distanceMatrix: np.ndarray, population: List[Individual], k: int) -> Individual:
    """k-tournament selection"""
    current_min = float('+inf')

    # To catch problems if all randomly chosen individuals have path length of infinity.
    best_ind = random.choice(population)

    for i in range(k):
        ind = random.choice(population)
        fit = fitness(distanceMatrix, ind.order)
        if fit < current_min:
            current_min = fit
            best_ind = ind
    return best_ind

def order_crossover(distanceMatrix: np.ndarray, parent1: Individual, parent2: Individual) -> Individual:
    length = (distanceMatrix.shape)[0]
    first = random.randint(0, length - 1)
    second = random.randint(0, length - 1)

    if second < first:
        first, second = second, first # Swap values

    new_order = np.zeros((length), dtype=np.int)
    new_order[first: second + 1] = parent1.order[first: second + 1]
    chosen_segment = set(parent1.order[first: second + 1]) # Transform the elements in the segment into a set -> O(1) lookup time
    position = (second + 1) % length
    for i in range(length):
        elem_of_parent_2 = parent2.order[(second + 1 + i) % length]
        if elem_of_parent_2 not in chosen_segment:
            new_order[position] = elem_of_parent_2
            position = (position + 1) % length
    beta = 2 * random.random() - 0.5 # Number between -0.5 and 3.5
    alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
    alpha = max(0.01, alpha)
    return Individual(distanceMatrix, new_order, alpha=alpha)

def add_elem_to_set(set_to_be_added: set, elem: int):
    if elem in set_to_be_added:
            set_to_be_added.remove(elem)
            set_to_be_added.add(-elem) # A minus denotes that an element is in both parents
    else:
        set_to_be_added.add(elem)

def add_neighbours(edge_table: List[set], parent: Individual, i: int, length: int):
    set_to_be_added = edge_table[parent.order[i]]
    elem = parent.order[(i + 1) % length]
    add_elem_to_set(set_to_be_added, elem)

    elem = parent.order[(i - 1)]
    add_elem_to_set(set_to_be_added, elem)

def construct_edge_table(parent1: Individual, parent2: Individual, length: int) -> List[set]:
    edge_table = [set() for _ in range(length)]
    # TODO: Maybe more efficient implementation?
    for i in range(length):
        add_neighbours(edge_table, parent1, i, length)
        add_neighbours(edge_table, parent2, i, length)
        
    return edge_table

def edge_crossover(distanceMatrix: np.ndarray, parent1: Individual, parent2: Individual) -> Individual:
    length = (distanceMatrix.shape)[0]

    # 1: Construct edge table
    edge_table = construct_edge_table(parent1, parent2, length)
    # 2: Pick an initial element at random and put it in the offspring
    node = random.randint(0, length - 1)	
    new_order = np.negative(np.ones((length), dtype=np.int))
    
    # 3: Set the variable 'node' to the randomly chosen element
    new_order[0] = node
    forward = True
    counter = 1
    while (counter != length):
        # 4: Remove all references to 'node' from the table
        for edge_set in edge_table:
            edge_set.discard(node)
            edge_set.discard(-node)
        # 5: Examine set for 'node'
        # 5a: If there is a common edge, pick that to be the next node
        current_set = edge_table[node]
        if len(current_set) != 0:
            double_edge_node = None
            for elem in current_set:
                if elem < 0:
                    double_edge_node = -elem
                    break
            if double_edge_node is not None:
                new_order[counter] = double_edge_node
                node = double_edge_node
                counter += 1
                continue
            
            # 5b: Otherwise, pick the entry in the set which itself has the shortest list. Ties are split randomly
            shortest = float('+inf')
            set_of_shortest_sets = set()
            for elem in current_set:
                len_elem = len(edge_table[elem])
                if len_elem < shortest:
                    shortest = len_elem
                    set_of_shortest_sets = set([elem])
                elif len_elem == shortest:
                    set_of_shortest_sets.add(elem)
            chosen_one = random.sample(set_of_shortest_sets, 1)[0] # Choose a random element from the shortest ones.
            new_order[counter] = chosen_one
            node = chosen_one
            counter += 1
            continue
        
        else:
            # 6a: In case of reaching an empty set, the other end of the offspring is examined for extension
            if forward:
                forward = False
                new_order[0: counter] = new_order[0: counter][::-1]
                node = new_order[counter - 1] # Set to other side
                continue
            # 6b: Otherwise, a new element is chosen at random
            # Reset direction again to forward
            forward = True
            new_order[0: counter] = new_order[0: counter][::-1]
            possibilities = set(range(length)) - set([elem for elem in new_order if elem >= 0])
            chosen_one = random.sample(possibilities, 1)[0] # Choose a random element from the possibilities
            new_order[counter] = chosen_one
            node = chosen_one
            counter += 1
    beta = 2 * random.random() - 0.5 # Number between -0.5 and 3.5
    alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
    alpha = max(0.01, alpha)
    return Individual(distanceMatrix, order=new_order, alpha=alpha)

def simple_edge_recombination(distanceMatrix: np.ndarray, parent1: Individual, parent2: Individual) -> Individual: # https://en.wikipedia.org/wiki/Edge_recombination_operator
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
    return Individual(distanceMatrix, order=np.array(new_order), alpha=alpha)

def mutation(individual: Individual):
    """Inversion mutation: randomly choose 2 indices and invert that subsequence."""   
    if random.random() < individual.alpha:
        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        individual.order[i: j] = individual.order[i: j][::-1]

def elimination(distanceMatrix: np.ndarray, population: List[Individual], offsprings: List[Individual], lambd: int) -> List[Individual]:
    """Mu + lambda elimination"""
    combined = population + offsprings  
    combined_with_fitness = {}
    for ind in combined:
        combined_with_fitness[ind] = fitness(distanceMatrix, ind.order)
    sorted_combined = [k for k, _ in sorted(combined_with_fitness.items(), key=lambda x:x[1], reverse=False)]
    return sorted_combined[:lambd]

def fitness_sharing_elimination(distanceMatrix: np.ndarray, population: List[Individual], offsprings: List[Individual], lambd: int) -> List[Individual]:
    """Mu + lambda elimination with fitness sharing.""" # TODO: Change this into another elimination procedure 
    all_individuals = population + offsprings 
    survivors = []
    for i in range(lambd):
        # beta_init = 1, because we want to count the individual itself (has itself not copied into survivors!)
        # Best possible approach to reduce computational cost --> Only recalculate fitness for the individuals that need recomputation 
        # (for most of them, their fitness will stay the same)
        fvals = fitness_sharing(distanceMatrix, all_individuals)
        idx = np.argmin(fvals)
        survivors.append(all_individuals[idx])
    return survivors

def fitness_sharing(distanceMatrix: np.ndarray, individuals: List[Individual]) -> List[float]:
    alpha = 1 # TODO: Put this parameter in the parameter class

    # TODO: Play with this 0.1. It denotes for example that for tour29, it will consider two solutions 
    # in each others neighbourhood if the Levenstein distance is less or equal than 2 (= 0.1 * 29 truncated). 
    sigma = int((distanceMatrix.shape)[0] * 0.1) 
    modified_fitness = np.zeros(len(individuals))
    for i, individual in enumerate(individuals):

        # Calculate all the Levenstein distances
        ds = []
        for j in range(len(individuals)):
            ds.append(levenshtein_distance(individual.order, individuals[j].order))


        # Note that x is in the population, so this also yields one time a + 1 for the beta (required!)
        one_plus_beta = 0
        for d in ds:
            if d <= sigma:
                one_plus_beta += 1 - (d / sigma) ** alpha 
        orig_fitness = fitness(distanceMatrix, individual.order)
        modified_fitness[i] = orig_fitness * one_plus_beta ** np.sign(orig_fitness)
    return modified_fitness

def build_cumulatives(distanceMatrix: np.ndarray, ind: Individual, length: int) -> Tuple[np.ndarray, np.ndarray]:
    cum_from_0_to_first = np.zeros((length))
    cum_from_second_to_end = np.zeros((length))
    cum_from_second_to_end[length - 1] = partial_fitness_one_value(distanceMatrix, frm=ind.order[-1], to=ind.order[0])
    for i in range(1, length - 1):
        cum_from_0_to_first[i] = cum_from_0_to_first[i - 1] \
            + partial_fitness_one_value(distanceMatrix, frm=ind.order[i-1], to=ind.order[i])
        cum_from_second_to_end[length - 1 - i] = cum_from_second_to_end[length - i] \
            + partial_fitness_one_value(distanceMatrix, frm=ind.order[length -1 - i], to=ind.order[length - i])
    return cum_from_0_to_first, cum_from_second_to_end

def local_search_operator_2_opt(distanceMatrix: np.ndarray, ind: Individual): # In-place
    """Local search operator, which makes use of 2-opt. Swap two edges within a cycle."""
    best_fitness = fitness(distanceMatrix, ind.order)
    length = len(ind.order)
    best_combination = (0, 0)

    cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, ind, length)
    if cum_from_second_to_end[-1] > 10_000_000:
        return

    for first in range(1, length - 2):
        fit_first_part = cum_from_0_to_first[first-1]
        if fit_first_part > 10_000_000 or fit_first_part > best_fitness:
            break
        fit_middle_part = 0.0
        for second in range(first + 2, length):
            fit_middle_part += partial_fitness_one_value(distanceMatrix, frm=ind.order[second-1], to=ind.order[second-2])
            if fit_middle_part > 10_000_000:
                break
            
            fit_last_part = cum_from_second_to_end[second]
            if fit_last_part > 10_000_000:
                continue

            bridge_first = partial_fitness_one_value(distanceMatrix, frm=ind.order[first-1], to=ind.order[second-1])
            bridge_second = partial_fitness_one_value(distanceMatrix, frm=ind.order[first], to=ind.order[second])
            temp = fit_first_part + fit_middle_part
            new_fitness = temp + fit_last_part + bridge_first + bridge_second
            if temp > best_fitness:
                continue
            
            if new_fitness < best_fitness:
                best_combination = (first, second)
                best_fitness = new_fitness
    best_first, best_second = best_combination
    if best_first == 0: # Initial individual was best
        return
    ind.order[best_first:best_second] = ind.order[best_first:best_second][::-1] # In-place

def get_new_order(first, second, ind):
    new_order = np.copy(ind.order)
    new_order[first:second] = new_order[first:second][::-1]
    return new_order

def swap_edges(ind: Individual, first: int, second: int) -> List[int]:
    """Swap two edges in a circle.
    Image the cycle (A, B, C, ..., Y, Z). If you swap the edges between C-D and Y-Z, 
    then the new cycle becomes (A, B, C, Y , X, W, V, ..., E, D, Z, A)."""  
    return np.concatenate((ind.order[0:first],
                           ind.order[second: - len(ind.order) + first - 1: -1],
                           ind.order[second + 1: len(ind.order)]))

def levenshtein_distance(token1, token2): # TODO: cite https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1
                
    return distances[len(token1)][len(token2)]


class r0652971:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename: str) -> float:
        # Read distance matrix from file.		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        p = Parameters(population_size=100, num_offsprings=100, k=5)

        population = initialization(distanceMatrix, p.population_size)
        best_fitness = float("+inf")
        for individual in population:
                fit = fitness(distanceMatrix, individual.order)
                if fit < best_fitness:
                    best_fitness = fit
        print("Best fitness after initialization:", best_fitness)

        best_fitnesses = []
        mean_fitnesses = []

        while(True): 

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
                offspring = edge_crossover(distanceMatrix, parent1, parent2)
                mutation(offspring) # In-place
                local_search_operator_2_opt(distanceMatrix, offspring) # In-place
                offsprings.append(offspring)
            
            # Mutation of the seed individuals
            for seed_individual in population:
                mutation(seed_individual) # In-place 

            population = elimination(distanceMatrix, population, offsprings, p.num_offsprings)
            # population = fitness_sharing_elimination(distanceMatrix, population, offsprings, p.num_offsprings)

            fitnesses = []
            best_fitness = float('+inf')
            for individual in population:
                fit = fitness(distanceMatrix, individual.order)
                fitnesses.append(fit)
                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = individual
            mean_fitness = sum(fitnesses) / len(fitnesses)
            timeLeft = self.reporter.report(mean_fitness, best_fitness, individual.order)
            if timeLeft < 0: 
                break
            # print(f"{it}: Mean fitness: {mean_fitness} \t Best fitness: {min(fitnesses)}")
            best_fitnesses.append(best_fitness)
            mean_fitnesses.append(mean_fitness)


        # Your code here.
        # plt.plot(x, mean_fitnesses, label='Mean fitnesses')
        # plt.plot(x, best_fitnesses, label='Best fitnesses')
        # plt.legend()
        # plt.show()
        # plt.savefig('mean_and_best_fitnesses.png')
        return best_fitness

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    problem = r0652971()
    problem.optimize('tours/tour1000.csv')

    pr.disable()
    pr.print_stats(sort="time")