import numpy as np
from pathlib import Path
import sys

try:
    from ..core.base_problem import BaseProblem
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.base_problem import BaseProblem


class ContinuousOptimizationProblem(BaseProblem):    
    def __init__(self, function_name, dimension = 10):
        super().__init__(problem_name=f"{function_name}-{dimension}D")
        self.problem_type = 'optimization'
        self.function_name = function_name
        self.dimension = dimension
        self.lower_bounds = None
        self.upper_bounds = None
        self.optimal_solution = None
        self.optimal_value = None
        self.function_evaluations = 0
        
    def load_data(self, **kwargs):
        self.metadata = {
            'function_name': self.function_name,
            'dimension': self.dimension,
            'bounds': (self.lower_bounds.tolist(), self.upper_bounds.tolist()),
            'optimal_value': self.optimal_value,
            'optimal_solution': self.optimal_solution.tolist() if self.optimal_solution is not None else None,
        }
    
    def evaluate(self, solution):
        if isinstance(solution, list):
            solution = np.array(solution)
        
        self.function_evaluations += 1
        return self.objective_function(solution)
    
    def objective_function(self, x):
        raise NotImplementedError("Subclasses must implement objective_function")
    
    def validate_solution(self, solution) -> bool:
        if isinstance(solution, list):
            solution = np.array(solution)
        
        if len(solution) != self.dimension:
            return False
        
        return np.all(solution >= self.lower_bounds) and np.all(solution <= self.upper_bounds)
    
    def get_bounds(self):
        return (self.lower_bounds, self.upper_bounds)
    
    def get_dimension(self):
        return self.dimension
    
    def reset_evaluations(self):
        self.function_evaluations = 0
    
    def compute_metrics(self, solution, additional_data=None):
        """
        additional_data: Optional dict with 'convergence_history', 'computation_time', etc.
        """
        metrics = super().compute_metrics(solution, additional_data)
        
        fitness = metrics['fitness']
        
        if self.optimal_value is not None:
            metrics['error'] = abs(fitness - self.optimal_value)
            metrics['relative_error'] = metrics['error'] / max(abs(self.optimal_value), 1e-10)
        
        metrics['function_evaluations'] = self.function_evaluations
        
        if additional_data and 'convergence_history' in additional_data:
            history = additional_data['convergence_history']
            if len(history) > 0:
                metrics['best_fitness'] = min(history)
                metrics['mean_fitness'] = np.mean(history[-100:])
                metrics['std_fitness'] = np.std(history[-100:])
                metrics['iterations'] = len(history)
                
                if len(history) > 10:
                    initial_fitness = history[0]
                    final_fitness = history[-1]
                    improvement = initial_fitness - final_fitness
                    if improvement > 0:
                        target_fitness = initial_fitness - 0.95 * improvement
                        converged_idx = next((i for i, f in enumerate(history) if f <= target_fitness), len(history))
                        metrics['convergence_speed'] = converged_idx
        
        if additional_data and 'computation_time' in additional_data:
            metrics['computation_time'] = additional_data['computation_time']
        
        return metrics


class RastriginFunction(ContinuousOptimizationProblem):
    """
    f(x) = 10n + Σ[x_i² - 10cos(2πx_i)]
    Domain: x_i ∈ [-5.12, 5.12]
    Global minimum: f(0, 0, ..., 0) = 0
    """
    def __init__(self, dimension = 10):
        super().__init__("Rastrigin", dimension)
        self.lower_bounds = np.full(dimension, -5.12)
        self.upper_bounds = np.full(dimension, 5.12)
        self.optimal_solution = np.zeros(dimension)
        self.optimal_value = 0.0
        self.load_data()
    
    def objective_function(self, x):
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


class AckleyFunction(ContinuousOptimizationProblem):
    """
    f(x) = -20exp(-0.2√(1/n Σx_i²)) - exp(1/n Σcos(2πx_i)) + 20 + e
    Domain: x_i ∈ [-5, 5]
    Global minimum: f(0, 0, ..., 0) = 0
    """
    def __init__(self, dimension = 10):
        super().__init__("Ackley", dimension)
        self.lower_bounds = np.full(dimension, -5.0)
        self.upper_bounds = np.full(dimension, 5.0)
        self.optimal_solution = np.zeros(dimension)
        self.optimal_value = 0.0
        self.load_data()
    
    def objective_function(self, x):
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        
        return term1 + term2 + 20 + np.e


class RosenbrockFunction(ContinuousOptimizationProblem):
    """
    f(x) = Σ[100(x_{i+1} - x_i²)² + (1 - x_i)²]
    Domain: x_i ∈ [-5, 10]
    Global minimum: f(1, 1, ..., 1) = 0
    """
    
    def __init__(self, dimension = 10):
        super().__init__("Rosenbrock", dimension)
        self.lower_bounds = np.full(dimension, -5.0)
        self.upper_bounds = np.full(dimension, 10.0)
        self.optimal_solution = np.ones(dimension)
        self.optimal_value = 0.0
        self.load_data()
    
    def objective_function(self, x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class SphereFunction(ContinuousOptimizationProblem):
    """
    f(x) = Σ x_i²
    Domain: x_i ∈ [-5.12, 5.12]
    Global minimum: f(0, 0, ..., 0) = 0
    """
    
    def __init__(self, dimension = 10):
        super().__init__("Sphere", dimension)
        self.lower_bounds = np.full(dimension, -5.12)
        self.upper_bounds = np.full(dimension, 5.12)
        self.optimal_solution = np.zeros(dimension)
        self.optimal_value = 0.0
        self.load_data()
    
    def objective_function(self, x):
        return np.sum(x**2)



def create_benchmark_function(function_name, dimension = 10) -> ContinuousOptimizationProblem:
    function_map = {
        'rastrigin': RastriginFunction,
        'ackley': AckleyFunction,
        'rosenbrock': RosenbrockFunction,
        'sphere': SphereFunction,
    }
    
    function_name_lower = function_name.lower()
    if function_name_lower not in function_map:
        available = ', '.join(function_map.keys())
        raise ValueError(f"Unknown function '{function_name}'. Available: {available}")
    
    return function_map[function_name_lower](dimension)


def get_all_benchmark_functions(dimensions = [10, 20, 30]):
    functions = ['rastrigin', 'ackley', 'rosenbrock', 'sphere']
    problems = []
    
    for func_name in functions:
        for dim in dimensions:
            problems.append(create_benchmark_function(func_name, dim))
    
    return problems
