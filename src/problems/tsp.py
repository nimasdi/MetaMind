import numpy as np
from pathlib import Path
from typing import Tuple
import sys

try:
    from ..core.base_problem import BaseProblem
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.base_problem import BaseProblem


class TSPProblem(BaseProblem):
    KNOWN_OPTIMA = {
        'eil51': 426,
        'berlin52': 7542,
        'kroA100': 21282,
    }
    
    def __init__(self, instance_name = "custom", n_cities = None):
        super().__init__(problem_name=f"TSP-{instance_name}")
        self.problem_type = 'optimization'
        self.instance_name = instance_name
        self.n_cities = n_cities
        self.cities = None  # Will store city coordinates
        self.distance_matrix = None
        
        if instance_name.lower() in self.KNOWN_OPTIMA:
            self.optimal_value = self.KNOWN_OPTIMA[instance_name.lower()]
    
    def load_data(self, file_path = None, generate_random = False, random_seed = None, bounds = (0, 100)):
        if generate_random:
            self.generate_random_instance(random_seed, bounds)
        elif file_path:
            self.load_tsplib_file(file_path)
        else:
            raise ValueError("Must provide either file_path or set generate_random=True")
        
        self.compute_distance_matrix()
        
        self.metadata = {
            'n_cities': self.n_cities,
            'instance_name': self.instance_name,
            'has_optimal': self.optimal_value is not None,
            'edge_weight_type': 'EUC_2D'
        }
    
    def generate_random_instance(self, seed, bounds):
        if self.n_cities is None:
            raise ValueError("n_cities must be specified for random instances")
        
        if seed is not None:
            np.random.seed(seed)
        
        self.cities = np.random.uniform(bounds[0], bounds[1], size=(self.n_cities, 2))
        self.instance_name = f"random_{self.n_cities}"
        self.problem_name = f"TSP-{self.instance_name}"
    
    def load_tsplib_file(self, file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"TSP file not found: {file_path}")
        
        cities = []
        reading_coords = False
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Parse header information
                if line.startswith('NAME'):
                    self.instance_name = line.split(':')[1].strip()
                    self.problem_name = f"TSP-{self.instance_name}"
                elif line.startswith('DIMENSION'):
                    self.n_cities = int(line.split(':')[1].strip())
                elif line.startswith('NODE_COORD_SECTION'):
                    reading_coords = True
                    continue
                elif line.startswith('EOF') or line.startswith('DISPLAY_DATA_SECTION'):
                    break
                
                # Read coordinates
                if reading_coords and line and not line.startswith('NODE_COORD_SECTION'):
                    parts = line.split()
                    if len(parts) >= 3:
                        # Format: city_id x y
                        x, y = float(parts[1]), float(parts[2])
                        cities.append([x, y])
        
        self.cities = np.array(cities)
        
        if len(self.cities) != self.n_cities:
            print(f"Warning: Expected {self.n_cities} cities, found {len(self.cities)}")
            self.n_cities = len(self.cities)
    
    def compute_distance_matrix(self):
        n = self.n_cities
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((self.cities[i] - self.cities[j]) ** 2))
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist
    
    def evaluate(self, tour):
        if isinstance(tour, list):
            tour = np.array(tour)
        
        total_distance = 0.0
        n = len(tour)
        
        for i in range(n):
            from_city = tour[i]
            to_city = tour[(i + 1) % n]
            total_distance += self.distance_matrix[from_city, to_city]
        
        return total_distance
    
    def validate_solution(self, tour):
        if len(tour) != self.n_cities:
            return False
        
        unique_cities = set(tour)
        if len(unique_cities) != self.n_cities:
            return False
        
        if min(unique_cities) < 0 or max(unique_cities) >= self.n_cities:
            return False
        
        return True
    
    def get_dimension(self):
        return self.n_cities
    
    def get_distance(self, city_i, city_j):
        return self.distance_matrix[city_i, city_j]
    
    def get_nearest_neighbor_tour(self, start_city = 0):
        tour = [start_city]
        unvisited = set(range(self.n_cities)) - {start_city}
        current_city = start_city
        
        while unvisited:
            nearest = min(unvisited, key=lambda city: self.distance_matrix[current_city, city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current_city = nearest
        
        distance = self.evaluate(tour)
        return tour, distance
    
    def compute_metrics(self, solution, additional_data = None):
        metrics = super().compute_metrics(solution, additional_data)
        
        metrics['tour_length'] = metrics['fitness']
        
        nn_tour, nn_distance = self.get_nearest_neighbor_tour()
        metrics['improvement_over_nn'] = ((nn_distance - metrics['tour_length']) / nn_distance) * 100
        
        if additional_data:
            if 'computation_time' in additional_data:
                metrics['computation_time'] = additional_data['computation_time']
            if 'iterations' in additional_data:
                metrics['iterations'] = additional_data['iterations']
            if 'convergence_history' in additional_data:
                metrics['convergence_count'] = len(additional_data['convergence_history'])
        
        return metrics
    
    def get_city_coordinates(self):
        return self.cities.copy()
    
    def __repr__(self):
        opt_str = f", optimal={self.optimal_value}" if self.optimal_value else ""
        return f"TSPProblem(instance='{self.instance_name}', cities={self.n_cities}{opt_str})"


def load_tsplib_instance(instance_name: str, data_dir: str = "data/tsplib") -> TSPProblem:
    problem = TSPProblem(instance_name=instance_name)
    file_path = Path(data_dir) / f"{instance_name}.tsp"
    problem.load_data(file_path=str(file_path))
    return problem


def create_random_tsp(n_cities: int, seed = None, bounds: Tuple[float, float] = (0, 100)) -> TSPProblem:
    problem = TSPProblem(instance_name="random", n_cities=n_cities)
    problem.load_data(generate_random=True, random_seed=seed, bounds=bounds)
    return problem
