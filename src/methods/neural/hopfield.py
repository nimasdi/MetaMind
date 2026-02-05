import time
import numpy as np
import sys
from pathlib import Path

try:
    from ...core.base_method import BaseMethod
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.base_method import BaseMethod

import torch


class HopfieldNetwork(BaseMethod):
    
    PARAM_SPECS = {
        'max_iterations': {
            'default': 100,
            'type': int,
            'range': (50, 500),
            'description': 'Maximum number of update iterations'
        },
        'threshold': {
            'default': 0.0,
            'type': float,
            'description': 'Activation threshold for neurons'
        },
        'async_update': {
            'default': True,
            'type': bool,
            'description': 'Use asynchronous (True) or synchronous (False) update'
        },
        'energy_threshold': {
            'default': 1e-6,
            'type': float,
            'description': 'Convergence threshold for energy change'
        }
    }
    
    def __init__(self, **parameters):
        full_params = self.get_default_parameters()
        full_params.update(parameters)
        super().__init__(**full_params)
        
        self.weights = None
        self.n_neurons = None
        self.patterns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, problem_data, callback=None, **kwargs):
        self.start_time = time.time()
        self.log("Starting Hopfield Network training")
        
        if isinstance(problem_data, dict):
            patterns = problem_data.get('patterns', problem_data.get('X'))
        else:
            patterns = problem_data
            
        if patterns is None:
            raise ValueError("No patterns provided for training")
        
        if torch.is_tensor(patterns):
            patterns = patterns.cpu().numpy()
        
        patterns = np.array(patterns)
        
        if np.all(np.isin(patterns, [0, 1])):
            self.log("Converting patterns from {0,1} to {-1,1}")
            patterns = 2 * patterns - 1
        elif not np.all(np.isin(patterns, [-1, 1])):
            raise ValueError("Patterns must contain only binary values {0,1} or {-1,1}")
        
        self.patterns = patterns
        n_patterns, n_features = patterns.shape
        self.n_neurons = n_features
        
        self.log(f"Training with {n_patterns} patterns of dimension {n_features}")
        
        patterns_tensor = torch.tensor(patterns, dtype=torch.float32, device=self.device)
        
        self.weights = torch.zeros((n_features, n_features), dtype=torch.float32, device=self.device)
        
        for i, pattern in enumerate(patterns_tensor):
            self.weights += torch.outer(pattern, pattern)
            
        self.weights /= n_patterns
        
        self.weights.fill_diagonal_(0)
        
        self.log(f"Weight matrix computed: shape {self.weights.shape}")
        
        capacity = n_features / (2 * np.log(n_features))
        if n_patterns > capacity:
            self.log(f"Warning: Storing {n_patterns} patterns exceeds theoretical capacity (~{capacity:.1f})")
        
        self.results = {
            'n_patterns': n_patterns,
            'n_neurons': n_features,
            'weights': self.weights.cpu().numpy(),
            'storage_capacity': capacity,
            'patterns': patterns
        }
        
        self.end_time = time.time()
        self.log(f"Training completed in {self.end_time - self.start_time:.4f} seconds")
        
        return self
    
    def predict(self, X, return_energy=False):
        if self.weights is None:
            raise ValueError("Model must be trained before prediction")
        
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if np.all(np.isin(X, [0, 1])):
            X = 2 * X - 1
        
        n_samples = X.shape[0]
        retrieved_patterns = []
        energy_trajectories = []
        
        for i in range(n_samples):
            result = self.retrieve_pattern(X[i], return_energy=return_energy)
            
            if return_energy:
                pattern, energy_history = result
                retrieved_patterns.append(pattern)
                energy_trajectories.append(energy_history)
            else:
                retrieved_patterns.append(result)
        
        retrieved_patterns = np.array(retrieved_patterns)
        
        if return_energy:
            return retrieved_patterns, energy_trajectories
        return retrieved_patterns
    
    def retrieve_pattern(self, initial_state, return_energy=False):
        state = torch.tensor(initial_state, dtype=torch.float32, device=self.device)
        
        max_iterations = self.parameters['max_iterations']
        threshold = self.parameters['threshold']
        async_update = self.parameters['async_update']
        energy_threshold = self.parameters['energy_threshold']
        
        energy_history = []
        prev_energy = self.compute_energy(state)
        energy_history.append(prev_energy)
        
        for iteration in range(max_iterations):
            if async_update:
                state = self.async_update(state, threshold)
            else:
                state = self.sync_update(state, threshold)
            
            current_energy = self.compute_energy(state)
            energy_history.append(current_energy)
            
            energy_change = abs(current_energy - prev_energy)
            
            if energy_change < energy_threshold:
                self.convergence_history.append({
                    'iteration': iteration + 1,
                    'energy': current_energy,
                    'energy_change': energy_change
                })
                break
            
            prev_energy = current_energy
        
        retrieved = state.cpu().numpy()
        
        if return_energy:
            return retrieved, np.array(energy_history)
        return retrieved
    
    def async_update(self, state, threshold):
        state = state.clone()
        n = len(state)
        
        indices = torch.randperm(n, device=self.device)
        
        for idx in indices:
            activation = torch.dot(self.weights[idx], state)
            
            if activation > threshold:
                state[idx] = 1.0
            elif activation < threshold:
                state[idx] = -1.0
        
        return state
    
    def sync_update(self, state, threshold):
        activations = torch.matmul(self.weights, state)
        
        new_state = torch.where(activations > threshold, torch.tensor(1.0, device=self.device),torch.tensor(-1.0, device=self.device))
        
        mask = (activations == threshold)
        new_state = torch.where(mask, state, new_state)
        
        return new_state
    
    def compute_energy(self, state):
        energy = -0.5 * torch.dot(state, torch.matmul(self.weights, state))
        return energy.item()

