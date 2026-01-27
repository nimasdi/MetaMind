from pathlib import Path
import sys
import numpy as np
import random
import operator
import math
from functools import partial
from deap import base, creator, tools, gp
import time


try:
    from ...core.base_method import BaseMethod
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.base_method import BaseMethod



class GeneticProgramming(BaseMethod):
    """
    Genetic Programming for symbolic regression and function approximation.
    Uses tree-based representation to evolve mathematical expressions.
    """
    
    PARAM_SPECS = {
        'population_size': {
            'type': int,
            'default': 200,
            'range': (100, 1000),
            'description': 'Number of individuals in the population'
        },
        'generations': {
            'type': int,
            'default': 50,
            'range': (20, 200),
            'description': 'Number of generations to evolve'
        },
        'max_depth': {
            'type': int,
            'default': 6,
            'range': (3, 10),
            'description': 'Maximum depth of expression trees'
        },
        'crossover_rate': {
            'type': float,
            'default': 0.9,
            'range': (0.7, 0.95),
            'description': 'Probability of crossover operation'
        },
        'mutation_rate': {
            'type': float,
            'default': 0.1,
            'range': (0.05, 0.2),
            'description': 'Probability of mutation operation'
        },
        'function_set': {
            'type': list,
            'default': ['+', '-', '*', '/'],
            'description': 'Set of functions/operators to use in trees'
        },
        'terminal_set': {
            'type': list,
            'default': ['x', 'constants'],
            'description': 'Set of terminals (variables and constants)'
        },
        'parsimony_coefficient': {
            'type': float,
            'default': 0.001,
            'range': (0, 0.01),
            'description': 'Coefficient for penalizing tree complexity'
        }
    }
    
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.best_individual = None
        self.toolbox = None
        self.pset = None
        self._setup_primitives()
        
    def _setup_primitives(self):
        num_inputs = self.parameters.get('num_inputs', 1)
        self.pset = gp.PrimitiveSet("MAIN", num_inputs)
        
        def protected_div(left, right):
            try:
                return left / right if abs(right) > 1e-6 else 1.0
            except (ZeroDivisionError, OverflowError):
                return 1.0
        
        def protected_sqrt(x):
            return math.sqrt(abs(x))
        
        def protected_log(x):
            return math.log(abs(x)) if abs(x) > 1e-6 else 0.0
        
        def protected_exp(x):
            try:
                return math.exp(min(max(x, -100), 100))
            except OverflowError:
                return 1.0
        
        def pow2(x):
            return x**2
        
        def pow3(x):
            return x**3
        
        function_set = self.parameters.get('function_set', ['+', '-', '*', '/'])
        
        function_mapping = {
            '+': (operator.add, 2),
            '-': (operator.sub, 2),
            '*': (operator.mul, 2),
            '/': (protected_div, 2),
            'sin': (math.sin, 1),
            'cos': (math.cos, 1),
            'tan': (math.tan, 1),
            'exp': (protected_exp, 1),
            'log': (protected_log, 1),
            'sqrt': (protected_sqrt, 1),
            'abs': (abs, 1),
            'neg': (operator.neg, 1),
            'pow2': (pow2, 1),
            'pow3': (pow3, 1),
        }
        
        for func_name in function_set:
            if func_name in function_mapping:
                func, arity = function_mapping[func_name]
                self.pset.addPrimitive(func, arity, name=func_name)
        
        terminal_set = self.parameters.get('terminal_set', ['x', 'constants'])
        if 'constants' in terminal_set:
            self.pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1))
        
        for i in range(num_inputs):
            self.pset.renameArguments(**{f'ARG{i}': f'x{i}'})
    
    def _setup_deap(self):
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        max_depth = self.parameters.get('max_depth', 6)
        
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=max_depth)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth+2))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth+2))
    
    def _evaluate_individual(self, individual, X, y):
        try:
            func = self.toolbox.compile(expr=individual)
            
            predictions = []
            for x_vals in X:
                if len(x_vals.shape) == 0:  # scalar
                    result = func(x_vals)
                else:
                    result = func(*x_vals)
                predictions.append(result)
            
            predictions = np.array(predictions)
            
            mse = np.mean((predictions - y) ** 2)
            
            # Add parsimony pressure (penalize complex trees)
            parsimony = self.parameters.get('parsimony_coefficient', 0.001)
            complexity_penalty = parsimony * len(individual)
            
            fitness = mse + complexity_penalty
            
            return (fitness,)
        except Exception as e:
            return (1e10,)
    
    def fit(self, problem_data, **kwargs):
        self.start_time = time.time()
        self.log("Starting Genetic Programming")
        
        X = problem_data['X']
        y = problem_data['y']
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.parameters['num_inputs'] = X.shape[1]
        self._setup_primitives()
        
        self._setup_deap()
        
        self.toolbox.register("evaluate", self._evaluate_individual, X=X, y=y)
        
        population_size = self.parameters.get('population_size', 200)
        generations = self.parameters.get('generations', 50)
        crossover_rate = self.parameters.get('crossover_rate', 0.9)
        mutation_rate = self.parameters.get('mutation_rate', 0.1)
        
        population = self.toolbox.population(n=population_size)
        
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Track initial best fitness
        best_ind = tools.selBest(population, 1)[0]
        best_fitness = best_ind.fitness.values[0]
        self.convergence_history.append(best_fitness)
        
        self.log(f"Initialized population of {population_size} individuals")
        self.log(f"Initial best fitness: {best_fitness:.6f}")
        
        for gen in range(generations):
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            best_prev = tools.selBest(population, 1)[0]
            worst_idx = offspring.index(tools.selWorst(offspring, 1)[0])
            offspring[worst_idx] = best_prev
            
            population[:] = offspring
            
            best_ind = tools.selBest(population, 1)[0]
            best_fitness = best_ind.fitness.values[0]
            self.convergence_history.append(best_fitness)
            
            if gen % 10 == 0 or gen == generations - 1:
                self.log(f"Generation {gen}/{generations}: Best Fitness = {best_fitness:.6f}")
        
        self.best_individual = tools.selBest(population, 1)[0]
        
        best_func = self.toolbox.compile(expr=self.best_individual)
        
        predictions = []
        for x_vals in X:
            if len(x_vals.shape) == 0:
                result = best_func(x_vals)
            else:
                result = best_func(*x_vals)
            predictions.append(result)
        
        predictions = np.array(predictions)
        final_mse = np.mean((predictions - y) ** 2)
        
        self.end_time = time.time()
        
        self.results = {
            'best_individual': str(self.best_individual),
            'best_fitness': best_fitness,
            'final_mse': final_mse,
            'generations': generations,
            'tree_depth': self.best_individual.height,
            'tree_size': len(self.best_individual),
            'training_time': self.end_time - self.start_time,
            'predictions': predictions
        }
        
        self.log(f"Training completed in {self.results['training_time']:.2f}s")
        self.log(f"Best expression: {str(self.best_individual)}")
        self.log(f"Tree depth: {self.best_individual.height}, size: {len(self.best_individual)}")
        
        return self
    
    def predict(self, X):
        if self.best_individual is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        best_func = self.toolbox.compile(expr=self.best_individual)
        
        predictions = []
        for x_vals in X:
            if len(x_vals.shape) == 0:
                result = best_func(x_vals)
            else:
                result = best_func(*x_vals)
            predictions.append(result)
        
        return np.array(predictions)
    
    def get_expression(self):
        if self.best_individual is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return str(self.best_individual)
