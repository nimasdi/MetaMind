from pathlib import Path
import sys
import time
import random
import math
import numpy as np

try:
	from ...core.base_method import BaseMethod
except ImportError:
	sys.path.insert(0, str(Path(__file__).parent.parent.parent))
	from core.base_method import BaseMethod


class AntColonyOptimization(BaseMethod):

	PARAM_SPECS = {
		'n_ants': {'type': int, 'default': 50, 'range': (20, 200)},
		'max_iterations': {'type': int, 'default': 500, 'range': (100, 2000)},
		'alpha': {'type': float, 'default': 1.0, 'range': (0.5, 2.0)}, # pheromone importance
		'beta': {'type': float, 'default': 2.0, 'range': (1.0, 5.0)}, # heuristic importance
		'evaporation_rate': {'type': float, 'default': 0.5, 'range': (0.1, 0.9)},  # evaporation
		'q': {'type': float, 'default': 1.0}, # pheromone deposit factor
		'initial_pheromone': {'type': float, 'default': 0.1},
		'local_search': {'type': bool, 'default': True} # apply 2-opt improvement
	}

	def __init__(self, **parameters):
		super().__init__(**parameters)

	def _get_distance_matrix(self, problem):
		if hasattr(problem, 'distance_matrix') and problem.distance_matrix is not None:
			return problem.distance_matrix
		# Try to build from get_distance
		n = getattr(problem, 'n_cities', None) or getattr(problem, 'get_dimension', lambda: None)()
		if n is None:
			raise ValueError("Problem must provide `distance_matrix` or `n_cities`/`get_distance`")
		mat = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if i == j:
					mat[i, j] = 0.0
				else:
					mat[i, j] = problem.get_distance(i, j)
		return mat

	def _construct_solution(self, pheromone, heuristic, alpha, beta, start=None):
		n = pheromone.shape[0]
		if start is None:
			current = random.randrange(n)
		else:
			current = start

		tour = [current]
		unvisited = set(range(n))
		unvisited.remove(current)

		while unvisited:
			probs = []
			denom = 0.0
			for j in unvisited:
				denom += (pheromone[current, j] ** alpha) * (heuristic[current, j] ** beta)

			if denom == 0.0:
				# fallback: uniform random among unvisited
				next_city = random.choice(list(unvisited))
			else:
				for j in unvisited:
					num = (pheromone[current, j] ** alpha) * (heuristic[current, j] ** beta)
					probs.append((j, num / denom))

				# Roulette wheel selection
				r = random.random()
				cumulative = 0.0
				next_city = None
				for (city, p) in probs:
					cumulative += p
					if r <= cumulative:
						next_city = city
						break
				if next_city is None:
					next_city = probs[-1][0]

			tour.append(next_city)
			unvisited.remove(next_city)
			current = next_city

		return tour

	def _tour_length(self, tour, distance_matrix):
		length = 0.0
		n = len(tour)
		for i in range(n):
			a = tour[i]
			b = tour[(i + 1) % n]
			length += distance_matrix[a, b]
		return length

	def _two_opt(self, tour, distance_matrix, max_iterations=None):
		"""Apply 2-opt local search to improve tour."""
		best_tour = tour.copy()
		best_length = self._tour_length(best_tour, distance_matrix)
		improved = True
		iterations = 0
		n = len(tour)

		while improved and (max_iterations is None or iterations < max_iterations):
			improved = False
			for i in range(1, n - 2):
				for j in range(i + 1, n):
					if j - i == 1:
						continue
					# Calculate change in tour length
					a, b = best_tour[i - 1], best_tour[i]
					c, d = best_tour[j], best_tour[(j + 1) % n]
					
					old_dist = distance_matrix[a, b] + distance_matrix[c, d]
					new_dist = distance_matrix[a, c] + distance_matrix[b, d]
					
					if new_dist < old_dist:
						# Reverse segment
						best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
						best_length = best_length - old_dist + new_dist
						improved = True
						break
				if improved:
					break
			if improved:
				iterations += 1

		return best_tour, best_length

	def fit(self, problem_data, **kwargs):
		self.start_time = time.time()
		self.log('ACO started')

		params = self.parameters
		# fill defaults
		for k, v in self.get_default_parameters().items():
			params.setdefault(k, v)

		n_ants = params['n_ants']
		max_iter = params['max_iterations']
		alpha = params['alpha']
		beta = params['beta']
		evaporation_rate = params['evaporation_rate']
		q = params['q']
		init_tau = params['initial_pheromone']
		use_local_search = params['local_search']
		seed = params.get('seed', None)

		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)

		# Problem must be an object with evaluate() and distance info
		problem = problem_data
		distance_matrix = self._get_distance_matrix(problem)
		n_cities = distance_matrix.shape[0]

		# Heuristic: inverse distance, avoid div by zero
		heuristic = np.zeros_like(distance_matrix)
		with np.errstate(divide='ignore', invalid='ignore'):
			heuristic = 1.0 / (distance_matrix + 1e-10)
		np.fill_diagonal(heuristic, 0.0)

		pheromone = np.full((n_cities, n_cities), init_tau, dtype=float)

		best_tour = None
		best_length = float('inf')

		self.convergence_history = []

		for iteration in range(1, max_iter + 1):
			ants_tours = []
			ants_lengths = []

			for ant in range(n_ants):
				start = random.randrange(n_cities)
				tour = self._construct_solution(pheromone, heuristic, alpha, beta, start=start)
				length = self._tour_length(tour, distance_matrix)

				# Apply local search improvement if enabled
				if use_local_search:
					tour, length = self._two_opt(tour, distance_matrix, max_iterations=10)

				ants_tours.append(tour)
				ants_lengths.append(length)

				if length < best_length:
					best_length = length
					best_tour = tour.copy()

			# Evaporation
			pheromone *= (1.0 - evaporation_rate)

			# Deposit pheromone
			for tour, length in zip(ants_tours, ants_lengths):
				if length <= 0:
					continue
				deposit = q / length
				for i in range(len(tour)):
					a = tour[i]
					b = tour[(i + 1) % len(tour)]
					pheromone[a, b] += deposit
					pheromone[b, a] += deposit

			self.convergence_history.append(best_length)

			if iteration % max(1, max_iter // 10) == 0:
				self.log(f"Iteration {iteration}/{max_iter} best length: {best_length:.6f}")

		self.end_time = time.time()

		self.results = {
			'best_tour': best_tour,
			'best_length': best_length,
			'convergence_history': self.convergence_history.copy(),
			'n_iterations': max_iter,
			'n_ants': n_ants,
			'execution_time': self.end_time - self.start_time
		}

		self.log(f"ACO finished. Best length: {best_length:.6f} in {self.results['execution_time']:.2f}s")
		return self.results

