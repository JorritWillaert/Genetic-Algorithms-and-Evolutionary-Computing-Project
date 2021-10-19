"""
In the binary knapsack problem, you are given a set of objects (o1, o2, ..., on),
along with their values (v1, v2, ..., vn)
and their weights (w1, w2, ..., wn).
The goal is to maximize the total value of the selected objects, 
subject to a weight constraint.
"""
import numpy as np

class KnapsackProblem:
    def __init__(self, numObjects):
        self.values = [2 ** (np.random.randn()) for _ in range(numObjects)]
        self.weights = [2 ** (np.random.randn()) for _ in range(numObjects)]
        self.capacity = 0.25 * sum(self.weights) # 25 % of the total weight

    