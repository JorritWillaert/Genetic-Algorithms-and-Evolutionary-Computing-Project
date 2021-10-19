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

def main():
    kp = KnapsackProblem(10)
    ind = Individual(kp)
    print(f"Knapsack values: {kp.values}")
    print(f"Knapsack weights = {kp.weights}")
    print(f"Knapsack capacity = {kp.capacity}")
    print(f"Individual: {ind.order}")
    print(f"Objective value of the individual: {fitness(kp, ind)}")


if __name__ == '__main__':
    main()