import r0652971

import unittest
import numpy as np

class Tests(unittest.TestCase):

    def test_order_crossover(self):
        """Check if all elements are represented."""
        for i in range(100): # Try multiple times due to random initializations
            with open('tours/tour29.csv') as f:
                distanceMatrix = np.loadtxt(f, delimiter=",")
            
            offspring = r0652971.order_crossover(distanceMatrix, 
                            r0652971.Individual(distanceMatrix), 
                            r0652971.Individual(distanceMatrix))
            sorted_order = np.sort(offspring.order)
            for i in range((distanceMatrix.shape)[0]):
                self.assertEqual(sorted_order[i], i)

    def test_edge_crossover(self):
        """Check if all elements are represented."""
        for i in range(100):
            with open('tours/tour29.csv') as f:
                distanceMatrix = np.loadtxt(f, delimiter=",")
            
            offspring = r0652971.edge_crossover(distanceMatrix, 
                            r0652971.Individual(distanceMatrix),
                            r0652971.Individual(distanceMatrix))
            sorted_order = np.sort(offspring.order)
            for i in range((distanceMatrix.shape)[0]):
                self.assertEqual(sorted_order[i], i)
                

if __name__ == "__main__":
    unittest.main()
