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
    
class PSO(BaseMethod):

    PARAM_SPECS = {
        'n_particles': {'type': int, 'default': 50, 'range': (20, 200)},
        'max_iterations': {'type': int, 'default': 500, 'range': (100, 2000)},
        'w': {'type': float, 'default': 0.7, 'range': (0.4, 0.9)},  
        'c1': {'type': float, 'default': 1.5, 'range': (1.0, 2.5)},  
        'c2': {'type': float, 'default': 1.5, 'range': (1.0, 2.5)},  
        'w_decay': {'type': bool, 'default': True},  
        'velocity_clamp': {'type': float, 'default': 0.5, 'range': (0.1, 1.0)} 
    }
    
    def fit(self, problem_data, callback=None, **kwargs):
        self.start_time = time.time()
        self.log("PSO optimization started.")
        
        params = self.parameters
        n_particles = params['n_particles']
        max_iter = params['max_iterations']
        w = params['w']
        c1 = params['c1']
        c2 = params['c2']
        w_decay = params['w_decay']
        v_clamp_frac = params['velocity_clamp']
        
        if hasattr(problem_data, 'evaluate'):
            objective_function = problem_data.evaluate
        elif hasattr(problem_data, 'get') and callable(problem_data.get):
            objective_function = problem_data.get('objective_function')
        else:
            self.end_time = time.time()
            raise ValueError("Problem must have an 'evaluate' method or be a dict with 'objective_function'.")
        
        if hasattr(problem_data, 'get_bounds'):
            bounds_tuple = problem_data.get_bounds()
            if isinstance(bounds_tuple, tuple) and len(bounds_tuple) == 2:
                lower_bounds, upper_bounds = bounds_tuple
                bounds = list(zip(lower_bounds, upper_bounds))
            else:
                bounds = bounds_tuple
        elif hasattr(problem_data, 'get') and callable(problem_data.get):
            bounds = problem_data.get('bounds')
        else:
            self.end_time = time.time()
            raise ValueError("Problem must have a 'get_bounds' method or be a dict with 'bounds'.")
        
        if not callable(objective_function) or not bounds:
            self.end_time = time.time()
            raise ValueError("Could not extract callable objective_function and bounds from problem.")
        
        n_dimensions = len(bounds)
        search_ranges = np.array([b[1] - b[0] for b in bounds])
        v_max = v_clamp_frac * search_ranges
        
        self.log(f"Dimensions: {n_dimensions}, Particles: {n_particles}, Max Iterations: {max_iter}")

        particles = []
        gbest_value = float('inf')
        gbest_position = None
        
        for _ in range(n_particles):
            position = np.array([random.uniform(b[0], b[1]) for b in bounds])
            velocity = np.array([random.uniform(-v_max[i], v_max[i]) for i in range(n_dimensions)])
            
            fitness = objective_function(position)
            
            particle = {
                'position': position,
                'velocity': velocity,
                'pbest_position': position.copy(),
                'pbest_value': fitness,
                'fitness': fitness
            }
            
            # Actualizar mejor global
            if fitness < gbest_value:
                gbest_value = fitness
                gbest_position = position.copy()
                
            particles.append(particle)
            self.convergence_history.append(gbest_value)
            
        self.log(f"Initialization complete. Initial best fitness: {gbest_value:.4e}")

        for t in range(1, max_iter + 1):
            
            current_w = w
            if w_decay:
                current_w = w * (1 - (t / max_iter) * (1 - 0.4 / w)) 
            
            for i in range(n_particles):
                p = particles[i]
                r1, r2 = random.random(), random.random()
                
                cognitive_component = c1 * r1 * (p['pbest_position'] - p['position'])
                social_component = c2 * r2 * (gbest_position - p['position'])
                
                p['velocity'] = (current_w * p['velocity'] + 
                                 cognitive_component + 
                                 social_component)
                
                for d in range(n_dimensions):
                    if abs(p['velocity'][d]) > v_max[d]:
                        p['velocity'][d] = np.sign(p['velocity'][d]) * v_max[d]

                p['position'] += p['velocity']
                
                for d in range(n_dimensions):
                    min_bound, max_bound = bounds[d]
                    if p['position'][d] < min_bound:
                        p['position'][d] = min_bound
                        p['velocity'][d] = 0 
                    elif p['position'][d] > max_bound:
                        p['position'][d] = max_bound
                        p['velocity'][d] = 0 
                        
                p['fitness'] = objective_function(p['position'])
                
                if p['fitness'] < p['pbest_value']:
                    p['pbest_value'] = p['fitness']
                    p['pbest_position'] = p['position'].copy()
                    
                    if p['fitness'] < gbest_value:
                        gbest_value = p['fitness']
                        gbest_position = p['position'].copy()
            
            self.convergence_history.append(gbest_value)

            if callback:
                callback({
                    'method': 'PSO',
                    'iteration': t,
                    'max_iterations': max_iter,
                    'global_best_fitness': gbest_value,
                })
            
            if t % (max_iter // 10 or 1) == 0:
                self.log(f"Iteration {t}/{max_iter}: Best Fitness = {gbest_value:.4e}")

        self.end_time = time.time()
        self.results = {
            'best_fitness': gbest_value,
            'best_position': gbest_position.tolist() if gbest_position is not None else None,
            'best_solution': gbest_position.tolist() if gbest_position is not None else None,  # Alias for consistency
            'iterations_run': max_iter,
            'elapsed_time': self.end_time - self.start_time
        }
        self.log(f"PSO optimization finished. Final best fitness: {gbest_value:.4e}")
        return self.results
