import r0652971

import unittest
import numpy as np

class Tests(unittest.TestCase):

    def __init__(self, methodName) -> None:
        super().__init__(methodName)
        with open('tours/tour29.csv') as f:
            self.distanceMatrix = np.loadtxt(f, delimiter=',')       

    def test_order_crossover(self):
        """Check if all elements are represented."""
        for i in range(1000): # Try multiple times due to random initializations            
            offspring = r0652971.order_crossover(self.distanceMatrix, 
                            r0652971.Individual(self.distanceMatrix), 
                            r0652971.Individual(self.distanceMatrix))
            sorted_order = np.sort(offspring.order)
            for i in range((self.distanceMatrix.shape)[0]):
                self.assertEqual(sorted_order[i], i)

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
