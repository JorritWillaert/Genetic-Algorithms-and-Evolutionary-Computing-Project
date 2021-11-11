import numpy as np
import random
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

""" A basic evolutionary algorithm. """
class eggholderEA:

	""" Initialize the evolutionary algorithm solver. """
	def __init__(self, fun):
		self.alpha = 0.05     		# Mutation probability
		self.lambdaa = 100     		# Population size
		self.mu = self.lambdaa * 2	# Offspring size
		self.k = 3            		# Tournament selection
		self.intMax = 500     		# Boundary of the domain, not intended to be changed.
		self.numIters = 20			# Maximum number of iterations
		self.objf = fun

	""" The main evolutionary algorithm loop. """
	def optimize( self, plotFun = lambda x : None):
		# Initialize population
		population = self.intMax * np.random.rand(self.lambdaa, 2)

		plotFun((population, self.intMax))
		for i in range(self.numIters):
			# The evolutionary algorithm
			start = time.time()
			selected = self.selection(population, self.k)
			offspring = self.crossover(selected)
			joinedPopulation = np.vstack((self.mutation(offspring, self.alpha), population))
			population = self.elimination(joinedPopulation, self.lambdaa)
			itT = time.time() - start

			# Show progress
			fvals = self.objf(population)
			meanObj = np.mean(fvals)
			bestObj = np.min(fvals)
			print(f'{itT: .2f}s:\t Mean fitness = {meanObj: .5f} \t Best fitness = {bestObj: .5f}')
			plotFun((population, self.intMax))
		print('Done')

	""" Perform k-tournament selection to select pairs of parents. """
	def selection(self, population, k):
		selected = np.zeros((self.mu, 2))
		for ii in range( self.mu ):
			ri = random.choices(range(np.size(population,0)), k = self.k)
			min = np.argmin( self.objf(population[ri, :]) )
			selected[ii,:] = population[ri[min],:]
		return selected

	""" Perform box crossover as in the slides. """
	def crossover(self, selected):
		weights = 3*np.random.rand(self.lambdaa,2) - 1
		offspring = np.zeros((self.lambdaa, 2))
		lc = lambda x, y, w: np.clip(x + w * (y-x), 0, self.intMax)
		for ii, _ in enumerate(offspring):
			offspring[ii,:] = lc(selected[2*ii, :], selected[2*ii+1, :], weights[ii, :])
		return offspring

	""" Perform mutation, adding a random Gaussian perturbation. """
	def mutation(self, offspring, alpha):
		ii = np.where(np.random.rand(np.size(offspring,0)) <= alpha)
		offspring[ii,:] = offspring[ii,:] + 10*np.random.randn(np.size(ii),2)
		offspring[ii,:] = np.clip(offspring[ii,:], 0, self.intMax)
		return offspring

	""" Eliminate the unfit candidate solutions. """
	def elimination(self, joinedPopulation, keep):
		fvals = self.objf(joinedPopulation)
		perm = np.argsort(fvals)
		survivors = joinedPopulation[perm[1:keep],:]
		return survivors


""" Compute the objective function at the vector of (x,y) values. """
def myfun(x):
	if np.size(x) == 2:
		x = reshape(x, (1,2))
	sas = np.sqrt(np.abs(x[:,0]+x[:,1]))
	sad = np.sqrt(np.abs(x[:,0]-x[:,1]))
	f = -x[:,1] * np.sin(sas) - x[:,0] * np.sin(sad)
	return f

"""
Make a 3D visualization of the optimization landscape and the location of the
given candidate solutions (x,y) pairs in input[0].
"""
def plotPopulation3D(input):
	population = input[0]

	x = np.arange(0,input[1],0.5)
	y = np.arange(0,input[1],0.5)
	X, Y = np.meshgrid(x, y)
	Z = myfun(np.transpose(np.vstack((X.flatten('F'),Y.flatten('F')))))
	Z = np.reshape(Z, (np.size(x), np.size(y)))

	fig = plt.gcf()
	fig.clear()
	ax = fig.gca(projection='3d')
	ax.scatter(population[:,0], population[:,1], myfun(population)+1*np.ones(population.shape[0]), c='r', marker='o')
	ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=1, antialiased=False, alpha=0.2)
	plt.pause(0.05)

"""
Make a 2D visualization of the optimization landscape and the location of the
given candidate solutions (x,y) pairs in input[0].
"""
def plotPopulation2D(input):
	population = input[0]

	x = np.arange(0,input[1],1)
	y = np.arange(0,input[1],1)
	X, Y = np.meshgrid(x, y)
	Z = myfun(np.transpose(np.vstack((X.flatten('F'),Y.flatten('F')))))
	Z = np.reshape(Z, (np.size(x), np.size(y)))

	# Determine location of minimum
	rowMin = np.min(Z, axis=0)
	minj = np.argmin(rowMin)
	colMin = np.min(Z, axis=1)
	mini = np.argmin(colMin)

	fig = plt.gcf()
	fig.clear()
	ax = fig.gca()
	ax.imshow(Z.T)
	ax.scatter(population[:,0], population[:,1], marker='o', color='r')
	ax.scatter(mini, minj, marker='*', color='yellow')
	plt.pause(0.05)


eggEA = eggholderEA(myfun)
eggEA.optimize(plotPopulation2D)
plt.show()
