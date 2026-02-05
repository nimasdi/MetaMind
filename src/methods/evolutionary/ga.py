from pathlib import Path
import random
import sys
import time
import numpy as np

try:
    from ...core.base_method import BaseMethod
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.base_method import BaseMethod


class GeneticAlgorithm(BaseMethod):
    PARAM_SPECS = {
        'population_size': {
            'type': int,
            'range': (50, 500),
            'default': 100
        },
        'generations': {
            'type': int,
            'range': (100, 2000),
            'default': 500
        },
        'crossover_rate': {
            'type': float,
            'range': (0.6, 0.95),
            'default': 0.8
        },
        'mutation_rate': {
            'type': float,
            'range': (0.01, 0.3),
            'default': 0.1
        },
        'selection': {
            'type': str,
            'options': ['tournament', 'roulette', 'rank'],
            'default': 'tournament'
        },
        'tournament_size': {
            'type': int,
            'range': (2, 10),
            'default': 3
        },
        'elitism': {
            'type': int,
            'range': (0, 10),
            'default': 2
        },
        'crossover_type': {
            'type': str,
            'options': ['pmx', 'ox', 'cx'],
            'default': 'pmx'
        }
    }

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.convergence_history = []

    def initialize_population(self, problem_size):
        population = []
        for _ in range(self.parameters['population_size']):
            individual = list(range(problem_size))
            random.shuffle(individual)
            population.append(individual)
        return population

    def evaluate_fitness(self, individual, problem_data):
        if hasattr(problem_data, 'evaluate'):
            return problem_data.evaluate(individual)
        raise NotImplementedError("Problem data must have an 'evaluate' method.")

    def selection_tournament(self, population, fitnesses):
        tournament_size = self.parameters['tournament_size']
        selected_indices = random.sample(range(len(population)), tournament_size)
        selected_fitnesses = [fitnesses[i] for i in selected_indices]
        winner_idx = selected_indices[np.argmin(selected_fitnesses)]
        return population[winner_idx]

    def selection_roulette(self, population, fitnesses):
        max_fitness = max(fitnesses)
        adjusted_fitnesses = [max_fitness - f + 1e-10 for f in fitnesses]
        total = sum(adjusted_fitnesses)
        probabilities = [f / total for f in adjusted_fitnesses]
        return random.choices(population, weights=probabilities, k=1)[0]

    def selection_rank(self, population, fitnesses):
        ranked_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        ranks = [rank + 1 for rank in range(len(fitnesses))]
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        selected_idx = random.choices(range(len(population)), weights=probabilities, k=1)[0]
        return population[selected_idx]

    def select_parent(self, population, fitnesses):
        selection = self.parameters['selection']
        if selection == 'tournament':
            return self.selection_tournament(population, fitnesses)
        elif selection == 'roulette':
            return self.selection_roulette(population, fitnesses)
        elif selection == 'rank':
            return self.selection_rank(population, fitnesses)
        else:
            raise ValueError(f"Unknown selection method: {selection}")

    def crossover_pmx(self, parent1, parent2):
        size = len(parent1)
        child1 = [None] * size
        child2 = [None] * size

        cx_point1, cx_point2 = sorted(random.sample(range(size), 2))

        child1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        child2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]

        mapping1 = {}
        mapping2 = {}
        for i in range(cx_point1, cx_point2):
            mapping1[parent2[i]] = parent1[i]
            mapping2[parent1[i]] = parent2[i]

        used1 = set(child1)
        used2 = set(child2)

        for i in range(size):
            if child1[i] is None:
                val = parent2[i]
                while val in mapping1:
                    val = mapping1[val]
                if val in used1:
                    # This should not happen in theory, but safety
                    for v in range(size):
                        if v not in used1:
                            val = v
                            break
                child1[i] = val
                used1.add(val)

        for i in range(size):
            if child2[i] is None:
                val = parent1[i]
                while val in mapping2:
                    val = mapping2[val]
                if val in used2:
                    for v in range(size):
                        if v not in used2:
                            val = v
                            break
                child2[i] = val
                used2.add(val)

        return child1, child2

    def crossover_ox(self, parent1, parent2):
        size = len(parent1)
        child1, child2 = [None] * size, [None] * size
        cx_point1, cx_point2 = sorted(random.sample(range(size), 2))

        child1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        child2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]

        def fill_child(child, parent, start_idx):
            pos = start_idx
            for i in range(start_idx, size):
                if parent[i] not in child:
                    while child[pos] is not None:
                        pos = (pos + 1) % size
                    child[pos] = parent[i]
            for i in range(start_idx):
                if parent[i] not in child:
                    while child[pos] is not None:
                        pos = (pos + 1) % size
                    child[pos] = parent[i]

        fill_child(child1, parent2, cx_point2)
        fill_child(child2, parent1, cx_point2)

        return child1, child2

    def crossover_cx(self, parent1, parent2):
        size = len(parent1)
        child1, child2 = [None] * size, [None] * size
        used = [False] * size

        i = 0
        while i < size and not all(used):
            if used[i]:
                i += 1
                continue
                
            cycle = []
            current = i
            while not used[current]:
                cycle.append(current)
                used[current] = True
                current = parent2.index(parent1[current])

            for idx in cycle:
                child1[idx] = parent1[idx]
                child2[idx] = parent2[idx]

            i += 1

        return child1, child2

    def crossover(self, parent1, parent2):
        crossover_type = self.parameters['crossover_type']
        if crossover_type == 'pmx':
            return self.crossover_pmx(parent1, parent2)
        elif crossover_type == 'ox':
            return self.crossover_ox(parent1, parent2)
        elif crossover_type == 'cx':
            return self.crossover_cx(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover type: {crossover_type}")

    def mutate(self, individual, problem_size):
        mutation_rate = self.parameters['mutation_rate']
        individual = individual.copy()
        for i in range(problem_size):
            if random.random() < mutation_rate:
                j = random.randint(0, problem_size - 1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    def apply_elitism(self, population, fitnesses):
        elitism_count = self.parameters['elitism']
        if elitism_count == 0:
            return []
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        return [population[i].copy() for i in sorted_indices[:elitism_count]]

    def fit(self, problem_data, callback=None, **kwargs):
        problem_size = kwargs.get('problem_size', len(problem_data))
        self.start_time = time.time()
        self.population = self.initialize_population(problem_size)
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.convergence_history = []


        self.log("Genetic Algorithm started...")

        for generation in range(self.parameters['generations']):

            fitnesses = [self.evaluate_fitness(ind, problem_data) for ind in self.population]
            current_best = min(fitnesses)
            current_best_idx = fitnesses.index(current_best)

            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_individual = self.population[current_best_idx].copy()
                self.convergence_history.append(self.best_fitness)

            self.fitness_history.append(current_best)

            if generation % max(1, self.parameters['generations'] // 10) == 0:
                self.log(f"Generation {generation}, Best Fitness: {current_best:.6f}")

            new_population = []

            elitism_count = self.parameters['elitism']
            if elitism_count > 0:
                sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
                elites = [self.population[i].copy() for i in sorted_indices[:elitism_count]]
                new_population.extend(elites)

            while len(new_population) < self.parameters['population_size']:
                parent1 = self.select_parent(self.population, fitnesses)
                parent2 = self.select_parent(self.population, fitnesses)

                if random.random() < self.parameters['crossover_rate']:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                child1 = self.mutate(child1, problem_size)
                child2 = self.mutate(child2, problem_size)

                new_population.append(child1)

                if len(new_population) < self.parameters['population_size']:
                    new_population.append(child2)

            if len(new_population) != self.parameters['population_size']:
                raise ValueError(
                    f"Population size mismatch: expected {self.parameters['population_size']}, got {len(new_population)}"
                )

            self.population = new_population

            if callback:
                callback({
                    'method': 'GeneticAlgorithm',
                    'iteration': generation + 1,
                    'max_iterations': self.parameters['generations'],
                    'current_fitness': current_best,
                    'best_fitness': self.best_fitness,
                    'current_generation_best': current_best,
                })

        final_fitnesses = [self.evaluate_fitness(ind, problem_data) for ind in self.population]
        final_best = min(final_fitnesses)
        final_best_idx = final_fitnesses.index(final_best)

        if final_best < self.best_fitness:
            self.best_fitness = final_best
            self.best_individual = self.population[final_best_idx].copy()

        self.end_time = time.time()

        self.results = {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'convergence_history': self.convergence_history.copy(),
            'total_generations': self.parameters['generations'],
            'population_size': self.parameters['population_size'],
            'final_population': self.population.copy(),
            'execution_time': self.end_time - self.start_time
        }

        self.log(f"Genetic Algorithm completed in {self.results['execution_time']:.2f}s")
        self.log(f"Best fitness: {self.best_fitness:.6f}")

        return self.results    