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
        'kroa100': 21282,
    }
    
    def __init__(self, instance_name = "custom", n_cities = None):
        super().__init__(problem_name=f"TSP-{instance_name}")
        self.problem_type = 'optimization'
        self.instance_name = instance_name
        self.n_cities = n_cities
        self.cities = None  # Will store city coordinates
        self.distance_matrix = None
        self.optimal_value = None
        
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
        
        if self.instance_name.lower() in self.KNOWN_OPTIMA:
            self.optimal_value = self.KNOWN_OPTIMA[self.instance_name.lower()]
    
    def compute_distance_matrix(self):
        n = self.n_cities
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((self.cities[i] - self.cities[j]) ** 2))
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist
    
    def evaluate(self, tour):
        arr = np.array(tour)

        # If a continuous vector (e.g., random-key encoding) is provided,
        # convert to a permutation by sorting (argsort). Otherwise, treat
        # the input as a permutation of city indices.
        if arr.dtype.kind in ('f', 'c'):
            perm = np.argsort(arr).astype(int)
        else:
            perm = arr.astype(int)

        if len(perm) != self.n_cities:
            raise ValueError(f"Tour length {len(perm)} does not match number of cities {self.n_cities}")

        total_distance = 0.0
        n = len(perm)

        for i in range(n):
            from_city = int(perm[i])
            to_city = int(perm[(i + 1) % n])
            total_distance += self.distance_matrix[from_city, to_city]

        return float(total_distance)
    
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

    def get_bounds(self):
        # for the pso we need this function
        if self.n_cities is None:
            raise ValueError("n_cities must be set before getting bounds")

        lower = np.zeros(self.n_cities)
        upper = np.ones(self.n_cities)
        return (lower, upper)
    
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
    
    def two_opt(self, tour):
        tour = list(tour)
        n = len(tour)
        improved = True
        best_distance = self.evaluate(tour)
        
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Try reversing the segment tour[i:j+1]
                    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    new_distance = self.evaluate(new_tour)
                    
                    if new_distance < best_distance:
                        tour = new_tour
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
        
        return tour, best_distance
    
    def get_lkh_estimation(self, max_iterations=1000, time_limit=60, num_starts=None, use_2opt=None):
        """
        Implementation of Iterated Local Search (ILS).
        Now includes 'num_starts' and 'use_2opt' as dummy arguments 
        to prevent errors from older scripts calling this method.
        """
        import time
        start_time = time.time()
        
        # 1. Start with a random tour or NN tour
        current_tour, current_dist = self.get_nearest_neighbor_tour()
        # Always use 2-opt in this improved version, regardless of the flag
        current_tour, current_dist = self.two_opt(current_tour)
        
        best_tour = list(current_tour)
        best_dist = current_dist
        
        print(f"  ILS Start: {best_dist:.2f}")
        
        no_improv_count = 0
        
        for i in range(max_iterations):
            if time.time() - start_time > time_limit:
                print(f"  Time limit reached during ILS.")
                break
                
            # 2. Perturbation (The "Kick")
            candidate_tour = self._double_bridge_kick(best_tour)
            
            # 3. Local Search
            candidate_tour, candidate_dist = self.two_opt(candidate_tour)
            
            # 4. Acceptance Criterion
            if candidate_dist < best_dist:
                best_dist = candidate_dist
                best_tour = list(candidate_tour)
                no_improv_count = 0
                print(f"  New best found at step {i}: {best_dist:.2f}")
            else:
                no_improv_count += 1
                
            # Early exit/restart if stuck
            if no_improv_count > 100:
                # Soft restart
                current_tour, _ = self.get_nearest_neighbor_tour(np.random.randint(0, self.n_cities))
                current_tour, current_dist = self.two_opt(current_tour)
                no_improv_count = 0

        print(f"  LKH-Approximation finished: {best_dist:.2f}")
        return best_tour, best_dist

    def _double_bridge_kick(self, tour):
        """
        Performs a 'Double-Bridge' move (a specific 4-opt move).
        This breaks 4 edges and reconnects them in a way 2-opt cannot immediately fix.
        Used to escape local optima.
        """
        n = len(tour)
        if n < 8:
            # Too small for double bridge, just shuffle
            new_tour = list(tour)
            np.random.shuffle(new_tour)
            return new_tour

        # We need 4 cut points: a < b < c < d
        # To avoid index errors, we pick 4 distinct sorted indices
        cuts = sorted(np.random.choice(n, 4, replace=False))
        a, b, c, d = cuts
        
        # A double bridge rearranges the tour segments:
        # Original:  [0...a] [a+1...b] [b+1...c] [c+1...d] [d+1...end]
        # Reordered: [0...a] [c+1...d] [b+1...c] [a+1...b] [d+1...end]
        
        # Note: In a list, slice [x:y] goes from x to y-1. 
        # So we need to be careful with indices.
        
        p1 = tour[:a+1]          # segment A
        p2 = tour[a+1:b+1]       # segment B
        p3 = tour[b+1:c+1]       # segment C
        p4 = tour[c+1:d+1]       # segment D
        p5 = tour[d+1:]          # segment E
        
        # Reconnect: A -> D -> C -> B -> E
        new_tour = p1 + p4 + p3 + p2 + p5
        
        return new_tour
    
    def solve_exact_branch_and_bound(self, time_limit=60):
        import time
        
        if self.n_cities > 30:
            raise ValueError(f"Exact solver not recommended for {self.n_cities} cities (> 25). Use LKH estimation instead.")
        
        print(f"  Running exact solver (branch-and-bound) for {self.n_cities} cities...")
        if self.n_cities >= 20:
            print(f"  Warning: This may take several minutes for 20+ cities")
        
        start_time = time.time()
        
        nn_tour, nn_distance = self.get_nearest_neighbor_tour()
        best_distance = nn_distance
        best_tour = nn_tour
        
        improved_tour, improved_distance = self.two_opt(nn_tour)
        if improved_distance < best_distance:
            best_distance = improved_distance
            best_tour = improved_tour
        
        print(f"  Initial upper bound: {best_distance:.2f}")
        
        nodes_explored = [0]
        nodes_pruned = [0]
        time_limit_reached = [False]
        
        def branch_and_bound(path, unvisited, current_distance):
            nonlocal best_distance, best_tour
            
            nodes_explored[0] += 1
            
            if nodes_explored[0] % 100000 == 0:
                if time.time() - start_time > time_limit:
                    if not time_limit_reached[0]:
                        time_limit_reached[0] = True
                        elapsed = time.time() - start_time
                        print(f"  Time limit reached at {elapsed:.1f}s! Returning best solution found...")
                    return True  # Signal to stop
            
            # If time limit already reached by another branch, stop immediately
            if time_limit_reached[0]:
                return True
            
            # Progress report every 1M nodes
            if nodes_explored[0] % 1000000 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {nodes_explored[0]/1e6:.1f}M nodes explored, "
                      f"{nodes_pruned[0]/1e6:.1f}M pruned, best={best_distance:.2f}, time={elapsed:.1f}s")
            
            # Base case: complete tour
            if not unvisited:
                total_distance = current_distance + self.distance_matrix[path[-1], path[0]]
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_tour = path[:]
                return False
            
            # Lower bound calculation with MST heuristic
            lower_bound = current_distance
            
            if unvisited:
                min_to_unvisited = min(self.distance_matrix[path[-1], city] for city in unvisited)
                lower_bound += min_to_unvisited
            
            if unvisited:
                min_to_start = min(self.distance_matrix[city, path[0]] for city in unvisited)
                lower_bound += min_to_start
            
            # Prune if lower bound exceeds best
            if lower_bound >= best_distance:
                nodes_pruned[0] += 1
                return False
            
            # Branch on nearest unvisited cities first
            current_city = path[-1]
            sorted_unvisited = sorted(unvisited, 
                                     key=lambda city: self.distance_matrix[current_city, city])
            
            for next_city in sorted_unvisited:
                # Check time limit in tight loop
                if time_limit_reached[0]:
                    return True
                    
                new_distance = current_distance + self.distance_matrix[current_city, next_city]
                
                # Prune if already exceeds best
                if new_distance >= best_distance:
                    nodes_pruned[0] += 1
                    continue
                
                path.append(next_city)
                unvisited.remove(next_city)
                
                stop = branch_and_bound(path, unvisited, new_distance)
                
                path.pop()
                unvisited.add(next_city)
                
                if stop:  # Time limit reached
                    return True
            
            return False
        
        initial_path = [0]
        initial_unvisited = set(range(1, self.n_cities))
        
        branch_and_bound(initial_path, initial_unvisited, 0.0)
        
        elapsed = time.time() - start_time
        
        if time_limit_reached[0]:
            print(f"  Exact solver stopped after {elapsed:.2f}s (time limit)")
            print(f"  Best solution found: {best_distance:.2f} (may not be optimal)")
        else:
            print(f"  Exact solver completed in {elapsed:.2f}s")
            print(f"  Optimal solution: {best_distance:.2f}")
        
        print(f"  Nodes explored: {nodes_explored[0]:,}, pruned: {nodes_pruned[0]:,}")
        
        return best_tour, best_distance
    
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
