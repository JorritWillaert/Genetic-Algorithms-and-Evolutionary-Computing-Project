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
		self.alpha = 0.05     																				# Mutation probability
		self.lambdaa = 100     																				# Population size
		self.mu = self.lambdaa * 2																			# Offspring size
		self.k = 3            																				# Tournament selection
		self.intMax = 500    												 								# Boundary of the domain, not intended to be changed.
		self.numIters = 20																					# Maximum number of iterations
		self.origfun = fun																					# The original unmodified fitness function
		#self.objf = lambda x, pop=None, beta_init = 0: self.shared_fitness_wrapper(fun, x, pop, beta_init)	# The objective function employed by the evolutionary algorithm
		self.objf = lambda x, pop: self.dominated_fitness_wrapper(fun, x, pop)

	def dominated_fitness_wrapper(self, fun, X, pop):
		dominated_counts = np.zeros(np.size(X, 0))
		xfvals = fun(X)
		fvals = fun(pop)
		for i, x in enumerate(X): # By how many are you dominated?
			for yfval in fvals: # This loop can probably be written more efficiently
				dominated_counts[i] += int(np.all(yfval <= xfvals[i,:])) # If so, then y is in all aspects better than x (true = 1, false = 0)
		return dominated_counts
	
	def shared_fitness_wrapper(self, fun, X, pop=None, beta_init=0):
		if pop is None:
			return fun(X)
		
		alpha = 1
		sigma = self.intMax * 0.1
		modified_fitness = np.zeros(X.shape[0])
		for i, x in enumerate(X):
			ds = self.distance(x, pop)

			# Note that x is in the population, so this also yields one time a + 1 for the beta (required!)
			one_plus_beta = beta_init
			for d in ds:
				if d <= sigma:
					one_plus_beta += 1 - (d / sigma) ** alpha 
			orig_fun = fun(x)
			modified_fitness[i] = orig_fun * one_plus_beta ** np.sign(orig_fun)
		return modified_fitness

	"""Calculate the Euclidean distances between one point (x) and an array of points (Y)."""
	def distance(self, x, Y):
		return np.linalg.norm(x - Y, axis=1) # Automatically broadcasts x


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
			population = self.selection(joinedPopulation, self.lambdaa) # For now let's just use k-tournament (it was first elimination or shared_elimination)
			itT = time.time() - start

			# Show progress
			fvals = self.origfun(population) # Note: changed this to origfun for multi-objective optimization
			meanObj = np.mean(fvals, axis=0) 
			bestObj = np.min(fvals, axis=0)
			print(f'{itT: .2f}s:\t Mean fitness = {meanObj[0]: .5f} \t Best fitness = {bestObj[0]: .5f}')
			plotFun((population, self.intMax))
		print('Done')

	""" Perform k-tournament selection to select pairs of parents. """
	def selection(self, population, k):
		selected = np.zeros((self.mu, 2))
		for ii in range( self.mu ):
			ri = random.choices(range(np.size(population,0)), k = self.k)
			min = np.argmin(self.objf(population[ri, :], population)) # Can also enable shared fitness optimization here by providing population as extra argument
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

	""" Eliminate the unfit candidate solutions using lambda + mu elimination. """
	def elimination(self, joinedPopulation, keep): # Note: lambda + mu elimination has a high selective pressure
		fvals = self.objf(joinedPopulation)
		perm = np.argsort(fvals)
		survivors = joinedPopulation[perm[1:keep],:]
		return survivors

	"""Lambda + mu elimination based on shared fitness values."""
	def shared_elimination(self, pop, keep):
		survivors = np.zeros((keep, 2))
		for i in range(keep):
			# beta_init = 1, because we want to count the individual itself (has itself not copied into survivors!)
			# Best possible approach to reduce computational cost --> Only recalculate fitness for the individuals that need recomputation 
			# (for most of them, their fitness will stay the same)
			fvals = self.objf(pop, survivors[0:i-1,:], beta_init=1) 
			idx = np.argmin(fvals)
			survivors[i,:] = pop[idx,:]
		return survivors


""" Compute the objective function at the vector of (x,y) values. """
def myfun(x):
	if np.size(x) == 2:
		x = np.reshape(x, (1,2))
	sas = np.sqrt(np.abs(x[:,0]+x[:,1]))
	sad = np.sqrt(np.abs(x[:,0]-x[:,1]))
	f = -x[:,1] * np.sin(sas) - x[:,0] * np.sin(sad)

	# You want to return the distance to the point (250, 250 as well)
	g = np.linalg.norm(x - np.array([250, 250]), axis=1)
	return np.transpose(np.vstack((f, g)))

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
	Z = np.reshape(Z[:,0], (np.size(x), np.size(y))) # Note: changed Z into Z[:, 0] for the multiobjective optimization.

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
	Z = np.reshape(Z[:, 0], (np.size(x), np.size(y))) # Note: changed Z into Z[:, 0] for the multiobjective optimization.

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