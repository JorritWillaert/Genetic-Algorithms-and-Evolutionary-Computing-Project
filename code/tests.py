from typing import List

import r0652971

import unittest
import numpy as np

class Tests(unittest.TestCase):

    def __init__(self, methodName) -> None:
        super().__init__(methodName)
        with open('tours/tour29.csv') as f:
            self.distanceMatrix = np.loadtxt(f, delimiter=',')  

    def test_a_solution(self):
        """Check a given solution."""
        solution = np.array([94,10,89,7,32,38,19,95,80,30,28,23,22,86,27,88,25,9,39,46,98,74,20,17,3,48,79,41,69,47,73,16,97,90,60,6,2,50,1,58,44,87,59,8,12,92,24,15,33,96,65,76,70,45,36,31,37,49,43,11,82,53,52,62,51,29,42,75,0,61,4,78,40,67,5,93,14,63,56,21,83,71,72,64,81,54,34,68,99,55,84,91,57,26,18,77,66,13,35,85])

        sorted_solution = np.sort(solution)

        for i in range(len(solution)):
            self.assertEqual(sorted_solution[i], i)   

    def test_order_crossover(self):
        """Check if all elements are represented."""
        for i in range(1000): # Try multiple times due to random initializations            
            offspring = r0652971.order_crossover(self.distanceMatrix, 
                            r0652971.Individual(self.distanceMatrix), 
                            r0652971.Individual(self.distanceMatrix))
            sorted_order = np.sort(offspring.order)
            for i in range((self.distanceMatrix.shape)[0]):
                self.assertEqual(sorted_order[i], i)

    def check_asserts(self, edge_table: List[set], element: int, edges: np.ndarray):
        for i in range(len(edges)):
            self.assertTrue(edges[i] in edge_table[element])     

    def test_edge_table(self):
        parent1 = r0652971.Individual(self.distanceMatrix, order=np.array([0,1,2,3,4,5,6,7,8]))
        parent2 = r0652971.Individual(self.distanceMatrix, order=np.array([8,2,6,7,1,5,4,0,3]))

        edge_table = r0652971.construct_edge_table(parent1, parent2, length=9)
        
        self.check_asserts(edge_table, 0, np.array([1,4,3,8]))
        self.check_asserts(edge_table, 1, np.array([0,2,5,7]))
        self.check_asserts(edge_table, 2, np.array([1,3,6,8]))
        self.check_asserts(edge_table, 3, np.array([0,2,4,8]))
        self.check_asserts(edge_table, 4, np.array([0,3,-5]))
        self.check_asserts(edge_table, 5, np.array([1,6,-4]))
        self.check_asserts(edge_table, 6, np.array([2,5,-7]))
        self.check_asserts(edge_table, 7, np.array([1,8,-6]))
        self.check_asserts(edge_table, 8, np.array([0,2,3,7]))

    def test_edge_crossover(self):
        """Check if all elements are represented."""
        for i in range(1000):        
            offspring = r0652971.edge_crossover(self.distanceMatrix, 
                            r0652971.Individual(self.distanceMatrix),
                            r0652971.Individual(self.distanceMatrix))
            sorted_order = np.sort(offspring.order)
            for i in range((self.distanceMatrix.shape)[0]):
                self.assertEqual(sorted_order[i], i)
                

if __name__ == "__main__":
    unittest.main()
