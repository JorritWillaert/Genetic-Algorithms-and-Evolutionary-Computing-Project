from numba import jit
import Reporter
import numpy as np
from typing import List, Tuple
import random
import multiprocessing
import psutil
import time
import warnings

INF = 10_000_000_000.0

class Parameters:
    def __init__(self, population_size: int, 
                       num_offsprings: int, 
                       k_selection: int, 
                       k_elimination: int, 
                       percentage_greedily: float, 
                       alpha: float, 
                       sigma_percentage: float):
        self.population_size = population_size
        self.num_offsprings = num_offsprings
        self.k_selection = k_selection
        self.k_elimination = k_elimination
        self.percentage_greedily = percentage_greedily
        self.alpha = alpha
        self.sigma_percentage = sigma_percentage

class Individual:
    def __init__(self, distanceMatrix: np.ndarray, 
                       order: np.ndarray=None, 
                       alpha: float=0.05):
        if order is None:
            self.order = np.random.permutation((distanceMatrix.shape)[0])
        else:
            self.order = order
        self.alpha = alpha
        self.length = (distanceMatrix.shape)[0]
        self.locally_optimal = False
        self.build_edges(self.order, self.length)

    def build_edges(self, order: np.ndarray, length: int):
        edges = [None] * length
        prev = order[0]
        for i in range(length):
            next = order[(i + 1) % length]
            edges[i] = (prev, next)
            prev = next
        self.edges = set(edges)

def initialization(distanceMatrix: np.ndarray, population_size: int, 
                   greedily_percentage: float) -> List[Individual]:
    greedily_number = int(greedily_percentage * population_size)
    legal_number =  population_size - greedily_number
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    manager = multiprocessing.Manager()
    L = manager.list()
    for _ in range(greedily_number):
        pool.apply_async(greedily_initialize_individual, args=(distanceMatrix, L))
    for _ in range(greedily_number, greedily_number + legal_number):
        pool.apply_async(initialize_legally, args=(distanceMatrix, L))
    pool.close()
    for _ in range(greedily_number + legal_number, population_size):
        L.append(Individual(distanceMatrix, alpha=max(0.04, 0.20+0.08*np.random.randn())))
    pool.join()
    #print("Initialization ended")
    return L

def initialize_legally(distanceMatrix: np.ndarray, L: list):
    length = (distanceMatrix.shape)[0]	
    i = 0
    start_time = time.time()
    while i != length:
        if time.time() - start_time > 2.0: # Don't spend more than 1 second initializing one individual
            print("Aborted initialization")
            L.append(Individual(distanceMatrix, alpha=max(0.04, 0.20+0.08*np.random.randn())))
        order = np.negative(np.ones((length), dtype=np.int))
        city = np.random.randint(0, length - 1)
        order[0] = city
        i = 1
        while i <= length:
            if i == length:
                if distanceMatrix[order[-1]][order[0]] == np.inf:
                    i = 0 # If returning to start yields a distance of infinity, start over again
                break 
            possibilities = set(range(length)) - set([elem for elem in order if elem >= 0])
            possibilities_legal = []
            for pos in possibilities:
                distance = distanceMatrix[city][pos]
                if distance < np.inf:
                    possibilities_legal.append(pos)
            if not possibilities_legal:
                break
            city = random.choice(possibilities_legal)
            order[i] = city
            i += 1
    L.append(Individual(distanceMatrix, order=order, alpha=max(0.04, 0.20+0.08*np.random.randn())))

def greedily_initialize_individual(distanceMatrix: np.ndarray, L: list):
    length = (distanceMatrix.shape)[0]	
    i = 0
    start_time = time.time()
    while i != length:
        if time.time() - start_time > 2.0: # Don't spend more than 1 second initializing one individual
            print("Aborted initialization")
            L.append(Individual(distanceMatrix, alpha=max(0.04, 0.20+0.08*np.random.randn())))
        order = np.negative(np.ones((length), dtype=np.int))
        city = np.random.randint(0, length - 1)
        order[0] = city
        i = 1
        while i <= length:
            if i == length:
                if distanceMatrix[order[-1]][order[0]] == np.inf:
                    i = 0 # If returning to start yields a distance of infinity, start over again
                break 
            possibilities = set(range(length)) - set([elem for elem in order if elem >= 0])
            min_distance = np.inf
            for pos in possibilities:
                distance = distanceMatrix[city][pos]
                if distance < min_distance:
                    min_distance = distance
                    new_city = pos
            if min_distance == np.inf:
                break
            city = new_city
            order[i] = city
            i += 1
    L.append(Individual(distanceMatrix, order=order, alpha=max(0.04, 0.20+0.08*np.random.randn())))

@jit(nopython=True)
def partial_fitness_one_value(distanceMatrix: np.ndarray, frm: int, to: int):
    distance = distanceMatrix[frm][to]
    if distance != np.inf:
        return distance
    return INF

@jit(nopython=True)
def fitness(distanceMatrix: np.ndarray, order: np.ndarray) -> float:
    fit = 0.0
    for i in range(len(order)):
        elem1 = order[i]
        elem2 = order[(i + 1) % len(order)]
        fit += distanceMatrix[elem1][elem2]
        if fit >= INF:
            return fit
    return fit

def selection(distanceMatrix: np.ndarray, population: List[Individual], k_selection: int, 
              all_fitnesses_hashmap: dict) -> Individual:
    """k-tournament selection"""
    current_min = float('+inf')

    # To catch problems if all randomly chosen individuals have path length of infinity.
    best_ind = random.choice(population)

    for i in range(k_selection):
        ind = random.choice(population)
        fit = all_fitnesses_hashmap.get(ind, -1)
        if fit == -1:
            fit = fitness(distanceMatrix, ind.order)
            all_fitnesses_hashmap[ind] = fit
        if fit < current_min:
            current_min = fit
            best_ind = ind
    return best_ind

@jit(nopython=True)
def order_crossover_jit(distanceMatrix: np.ndarray, parent1_order: np.ndarray, 
                        parent2_order: np.ndarray, new_order: np.ndarray) -> np.ndarray:
    length = (distanceMatrix.shape)[0]
    first = random.randint(0, length - 1)
    second = random.randint(0, length - 1)

    if second < first:
        first, second = second, first # Swap values

    new_order[first: second + 1] = parent1_order[first: second + 1]

    # Transform the elements in the segment into a set -> O(1) lookup time
    chosen_segment = set(parent1_order[first: second + 1])

    position = (second + 1) % length
    for i in range(length):
        elem_of_parent_2 = parent2_order[(second + 1 + i) % length]
        if elem_of_parent_2 not in chosen_segment:
            new_order[position] = elem_of_parent_2
            position = (position + 1) % length  
    return new_order

def order_crossover(distanceMatrix: np.ndarray, parent1: Individual, 
                    parent2: Individual) -> Individual:
    new_order = np.empty(((distanceMatrix.shape)[0]), dtype=np.int)
    new_order = order_crossover_jit(distanceMatrix, parent1.order, parent2.order, new_order)
    beta = 2 * random.random() - 0.5 # Number between -0.5 and 3.5
    alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
    alpha = max(0.04, alpha)
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

def construct_edge_table(parent1: Individual, parent2: Individual, 
                         length: int) -> List[set]:
    edge_table = [set() for _ in range(length)]
    for i in range(length):
        add_neighbours(edge_table, parent1, i, length)
        add_neighbours(edge_table, parent2, i, length)
        
    return edge_table

def edge_crossover(distanceMatrix: np.ndarray, parent1: Individual, 
                   parent2: Individual) -> Individual:
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
            
            # 5b: Otherwise, pick the entry in the set which itself has the shortest list. 
            # Ties are split randomly
            shortest = float('+inf')
            set_of_shortest_sets = set()
            for elem in current_set:
                len_elem = len(edge_table[elem])
                if len_elem < shortest:
                    shortest = len_elem
                    set_of_shortest_sets = set([elem])
                elif len_elem == shortest:
                    set_of_shortest_sets.add(elem)
            # Choose a random element from the shortest ones.
            chosen_one = random.sample(set_of_shortest_sets, 1)[0] 
            new_order[counter] = chosen_one
            node = chosen_one
            counter += 1
            continue
        
        else:
            # 6a: In case of reaching an empty set, the other end of the offspring 
            # is examined for extension
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

            # Choose a random element from the possibilities
            chosen_one = random.sample(possibilities, 1)[0] 
            new_order[counter] = chosen_one
            node = chosen_one
            counter += 1
    beta = 2 * random.random() - 0.5 # Number between -0.5 and 3.5
    alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
    alpha = max(0.04, alpha)
    return Individual(distanceMatrix, order=new_order, alpha=alpha)

# https://en.wikipedia.org/wiki/Edge_recombination_operator
def simple_edge_recombination(distanceMatrix: np.ndarray, parent1: Individual, 
                              parent2: Individual) -> Individual: 
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
            # Choose a random element from the shortest ones.
            chosen_one = random.sample(shortest_ones, 1)[0] 
            shortest_ones.remove(chosen_one)

        else:
            possible_ones = all_nodes - set(new_order)
            # Choose a random element from the possible ones.
            chosen_one = random.sample(possible_ones, 1)[0] 
            possible_ones.remove(chosen_one) 

        node = chosen_one
        new_order.append(node)
    beta = 2 * random.random() - 0.5 # Number between -0.5 and 3.5
    alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)
    alpha = max(0.04, alpha)
    return Individual(distanceMatrix, order=np.array(new_order), alpha=alpha)

def mutation(distanceMatrix: np.ndarray, individual: Individual, 
             all_fitnesses_hashmap: dict) -> Individual:
    """Inversion mutation: randomly choose 2 indices and invert that subsequence."""   
    if random.random() < individual.alpha: 
        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        new_order = np.copy(individual.order)
        new_order[i: j] = new_order[i: j][::-1]
        all_fitnesses_hashmap.pop(individual, None)
        return Individual(distanceMatrix, new_order, individual.alpha)
    return individual

def scramble_mutation(distanceMatrix: np.ndarray, individual: Individual, 
                      all_fitnesses_hashmap: dict) -> Individual:
    """Scramble mutation: randomly choose 2 indices and scramble that subsequence."""   
    if random.random() < individual.alpha:
        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        if j < i:
            i, j = j, i
        new_order = np.copy(individual.order)
        np.random.shuffle(new_order[i: j])
        all_fitnesses_hashmap.pop(individual, None)
        return Individual(distanceMatrix, new_order, individual.alpha)
    return individual

def elimination(distanceMatrix: np.ndarray, population: List[Individual], 
                offsprings: List[Individual], lambd: int) -> List[Individual]:
    """Mu + lambda elimination"""
    combined = population + offsprings  
    combined_with_fitness = {}
    for ind in combined:
        combined_with_fitness[ind] = fitness(distanceMatrix, ind.order)
    sorted_combined = [k for k, _ in sorted(combined_with_fitness.items(), key=lambda x:x[1], reverse=False)]
    return sorted_combined[:lambd]

def fitness_sharing_elimination(distanceMatrix: np.ndarray, population: List[Individual], 
                                offsprings: List[Individual], lambd: int,
                                all_distances_hashmap: dict) -> List[Individual]:
    """Mu + lambda elimination with fitness sharing."""
    all_individuals = population + offsprings 
    survivors = [] 
    fitnesses = np.array([fitness(distanceMatrix, order=ind.order) for ind in all_individuals])
    for i in range(lambd):
        # Best possible approach to reduce computational cost 
        # --> Only recalculate fitness for the individuals that need recomputation 
        # (for most of them, their fitness will stay the same)
        fvals = fitness_sharing(distanceMatrix, all_individuals, survivors[0:i-1], 
                                fitnesses, all_distances_hashmap)
        idx = np.argmin(fvals)
        survivors.append(all_individuals[idx])
        #del all_individuals[idx]
        #fitnesses = np.delete(fitnesses, idx)
    return survivors

def fitness_sharing_elimination_k_tournament(distanceMatrix: np.ndarray, population: List[Individual], 
                                             offsprings: List[Individual], lambd: int,
                                             k_elimination: int, alpha: float, 
                                             sigma_percentage: float, all_distances_hashmap: dict, 
                                             all_fitnesses_hashmap: dict) -> List[Individual]:
    """K-tournament fitness sharing elimination"""
    all_individuals = population + offsprings
    all_orig_individuals = all_individuals.copy()
    survivors = []
    fitnesses = np.empty((len(all_individuals)))
    for i, ind in enumerate(all_individuals):
        fit = all_fitnesses_hashmap.get(ind, -1)
        if fit == -1:
            fit = fitness(distanceMatrix, order=ind.order)
            all_fitnesses_hashmap[ind] = fit
        fitnesses[i] = fit 
    best_ind_idx = np.argmin(fitnesses)
    survivors.append(all_individuals[best_ind_idx])

    for i in range(lambd - 1):
        # Best possible approach to reduce computational cost 
        # --> Only recalculate fitness for the individuals that need recomputation 
        # (for most of them, their fitness will stay the same)
        fvals = fitness_sharing(distanceMatrix, all_individuals, 
                                survivors[0:i-1], alpha, sigma_percentage, 
                                fitnesses, all_distances_hashmap)
        
        current_min = INF
        # To catch problems if all randomly chosen individuals have path length of infinity.
        
        best_idx = 0
        
        for i in range(k_elimination - 1):
            idx = random.randint(0, len(fvals) - 1)
            fit = fvals[idx]
            if fit < current_min:
                current_min = fit
                best_idx = idx
        new_survivor = all_individuals[best_idx]
        survivors.append(new_survivor)
        del all_individuals[idx]
        fitnesses = np.delete(fitnesses, idx)
    for dead_ind in (set(all_orig_individuals).difference(survivors)):
        all_fitnesses_hashmap.pop(dead_ind, None)
    return survivors

def distance_from_to(first_ind: Individual, second_ind: Individual):
    edges_first = first_ind.edges
    edges_second = second_ind.edges
    intersection = edges_first.intersection(edges_second)
    num_edges_first = len(first_ind.edges)

    return num_edges_first - len(intersection)

def fitness_sharing(distanceMatrix: np.ndarray, population: List[Individual], 
                    survivors: np.ndarray, alpha: float, sigma_percentage: float, 
                    original_fits: np.ndarray, all_distances_hashmap: dict) -> np.ndarray:
    if not survivors:
        return original_fits

    sigma = int((distanceMatrix.shape)[0] * sigma_percentage) 
    
    distances = np.zeros((len(population), len(survivors)))
    for i in range(len(population)):
        for j in range(len(survivors)):
            distance1 = all_distances_hashmap.get((population[i], survivors[j]), -1)
            distance2 = all_distances_hashmap.get((survivors[j], population[i]), -1)
            if distance1 == -1 and distance2 == -1:
                distance = distance_from_to(population[i], survivors[j])
                all_distances_hashmap[(population[i], survivors[j])] = distance
            else:
                if distance1 != -1:
                    distance = distance1
                else:
                    distance = distance2
            distances[i][j] = distance
    shared_part = (1 - (distances / sigma) ** alpha)
    shared_part *= np.array(distances <= sigma)
    sum_shared_part = np.sum(shared_part, axis=1)
    shared_fitnesses = original_fits * sum_shared_part
    shared_fitnesses = np.where(np.isnan(shared_fitnesses), np.inf, 
                                shared_fitnesses)
    return shared_fitnesses

@jit(nopython=True)
def build_cumulatives(distanceMatrix: np.ndarray, order: np.ndarray, 
                      length: int) -> Tuple[np.ndarray, np.ndarray]:
    cum_from_0_to_first = np.zeros((length))
    cum_from_second_to_end = np.zeros((length))
    cum_from_second_to_end[length - 1] = partial_fitness_one_value(distanceMatrix, 
                                                                   frm=order[-1], 
                                                                   to=order[0])
    for i in range(1, length - 1):
        cum_from_0_to_first[i] = cum_from_0_to_first[i - 1] \
            + partial_fitness_one_value(distanceMatrix, frm=order[i-1], to=order[i])
        cum_from_second_to_end[length - 1 - i] = cum_from_second_to_end[length - i] \
            + partial_fitness_one_value(distanceMatrix, frm=order[length -1 - i], to=order[length - i])
    return cum_from_0_to_first, cum_from_second_to_end

@jit(nopython=True)
def local_search_operator_2_opt(distanceMatrix: np.ndarray, order: np.ndarray) -> Individual:
    """Local search operator, which makes use of 2-opt. Swap two edges within a cycle."""
    best_fitness = fitness(distanceMatrix, order)
    length = len(order)
    best_combination = (0, 0)

    cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, order, length)
    if cum_from_second_to_end[-1] > INF:
        return None

    for first in range(1, length - 2):
        fit_first_part = cum_from_0_to_first[first-1]
        if fit_first_part > INF or fit_first_part > best_fitness:
            break
        fit_middle_part = 0.0
        for second in range(first + 2, length):
            fit_middle_part += partial_fitness_one_value(distanceMatrix, 
                                                        frm=order[second-1], 
                                                        to=order[second-2])
            if fit_middle_part > INF:
                break
            
            fit_last_part = cum_from_second_to_end[second]
            if fit_last_part > INF:
                continue

            bridge_first = partial_fitness_one_value(distanceMatrix, 
                                                     frm=order[first-1], 
                                                     to=order[second-1])
            bridge_second = partial_fitness_one_value(distanceMatrix, 
                                                      frm=order[first], 
                                                      to=order[second])
            temp = fit_first_part + fit_middle_part
            if temp > best_fitness:
                continue
            new_fitness = temp + fit_last_part + bridge_first + bridge_second
            
            if new_fitness < best_fitness:
                best_combination = (first, second)
                best_fitness = new_fitness
    best_first, best_second = best_combination
    if best_first == 0: # Initial individual was best
        return None
    new_order = np.copy(order)
    new_order[best_first:best_second] = new_order[best_first:best_second][::-1]
    return new_order

def rotate_0_up_front(order: np.ndarray) -> np.ndarray:
    idx = np.where(order==0)
    return np.concatenate([order[int(idx[0]):], order[0:int(idx[0])]]) 


class r0652971:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename: str) -> float:
        # Read distance matrix from file.		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        p = Parameters(population_size=15, num_offsprings=15, k_selection=5, 
                       k_elimination=8, percentage_greedily=0.20, alpha=0.25, 
                       sigma_percentage=0.50)

        INF = np.nanmax(distanceMatrix[distanceMatrix != np.inf]) * (distanceMatrix.shape)[0]
        #print("Infinity is: " + str(INF))

        population = initialization(distanceMatrix, p.population_size, p.percentage_greedily)
        best_fitness = float("+inf")
        best_individual = population[0]
        for individual in population:
                fit = fitness(distanceMatrix, individual.order)
                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = individual
        #print("Best fitness after initialization:", best_fitness)
        
        all_distances_hashmap = {}
        all_fitnesses_hashmap = {}
        num = 0
        best_prev_fitness = np.inf


        while(True): 

            # To prevent thrashing (especially because RAM size of the testing framework is unknown) 
            if (psutil.virtual_memory()[2] > 95.0):
                all_distances_hashmap = {} 
                #all_fitnesses_hashmap = {}

            offsprings = []
            #count = 0
            for offspring in range(p.num_offsprings):
                parent1 = selection(distanceMatrix, population, p.k_selection, all_fitnesses_hashmap)
                parent2 = selection(distanceMatrix, population, p.k_selection, all_fitnesses_hashmap)
                offspring = order_crossover(distanceMatrix, parent1, parent2)
                offspring = mutation(distanceMatrix, offspring, all_fitnesses_hashmap)
                if not offspring.locally_optimal:
                    new_order = local_search_operator_2_opt(distanceMatrix, offspring.order)
                    if new_order is not None:
                        all_fitnesses_hashmap.pop(offspring, None)
                        offspring = Individual(distanceMatrix, new_order, offspring.alpha)
                    else:
                        offspring.locally_optimal = True
                offsprings.append(offspring)

            # Mutation of the seed individuals
            for i, seed_individual in enumerate(population):
                if seed_individual == best_individual:
                    continue
                population[i] = mutation(distanceMatrix, seed_individual, all_fitnesses_hashmap) 

            # population = elimination(distanceMatrix, population, offsprings, p.num_offsprings)
            population = fitness_sharing_elimination_k_tournament(distanceMatrix, population, 
                                                                  offsprings, p.num_offsprings, 
                                                                  p.k_elimination, p.alpha, 
                                                                  p.sigma_percentage, 
                                                                  all_distances_hashmap, 
                                                                  all_fitnesses_hashmap)

            fitnesses = []
            best_fitness = float('+inf')
            for individual in population:
                fit = all_fitnesses_hashmap.get(individual, -1)
                if fit == -1:
                    fit = fitness(distanceMatrix, individual.order)
                    all_fitnesses_hashmap[individual] = fit
                fitnesses.append(fit)
                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = individual
            mean_fitness = sum(fitnesses) / len(fitnesses)
            if num >= 100 or best_fitness < best_prev_fitness:
                report_order = rotate_0_up_front(best_individual.order)
                timeLeft = self.reporter.report(mean_fitness, best_fitness, report_order)
                if timeLeft < 0: 
                    break
                best_prev_fitness = best_fitness
                num = 0
            num += 1

        return best_fitness

if __name__ == "__main__":
    problem = r0652971()
    problem.optimize('tours/tour750.csv')