import numpy as np
import time
import sys
from pathlib import Path

try:
    from ...core.base_method import BaseMethod
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.base_method import BaseMethod


class SOM(BaseMethod):
    PARAM_SPECS = {
        'map_size': {
            'type': tuple,
            'default': (10, 10),
            'description': 'Size of the map grid (height, width)'
        },
        'learning_rate_initial': {
            'type': float,
            'default': 0.5,
            'range': (0.1, 1.0),
            'description': 'Initial learning rate'
        },
        'learning_rate_final': {
            'type': float,
            'default': 0.01,
            'description': 'Final learning rate'
        },
        'neighborhood_initial': {
            'type': float,
            'default': 5.0,
            'description': 'Initial neighborhood radius'
        },
        'max_epochs': {
            'type': int,
            'default': 1000,
            'range': (500, 5000),
            'description': 'Maximum number of training epochs'
        },
        'topology': {
            'type': str,
            'default': 'rectangular',
            'options': ['rectangular', 'hexagonal'],
            'description': 'Topology of the map: rectangular or hexagonal'
        }
    }
    
    def __init__(self, **parameters):
        super().__init__(**parameters)
        
        self.map_size = self.parameters.get('map_size', self.PARAM_SPECS['map_size']['default'])
        self.learning_rate_initial = self.parameters.get(
            'learning_rate_initial', 
            self.PARAM_SPECS['learning_rate_initial']['default']
        )
        self.learning_rate_final = self.parameters.get(
            'learning_rate_final',
            self.PARAM_SPECS['learning_rate_final']['default']
        )
        self.neighborhood_initial = self.parameters.get(
            'neighborhood_initial',
            self.PARAM_SPECS['neighborhood_initial']['default']
        )
        self.max_epochs = self.parameters.get(
            'max_epochs',
            self.PARAM_SPECS['max_epochs']['default']
        )
        self.topology = self.parameters.get(
            'topology',
            self.PARAM_SPECS['topology']['default']
        )
        
        self.weights = None
        self.map_positions = None
        self.neuron_activations = None
        self.data_mean = None
        self.data_std = None
        self.bmu_history = []  
        self.quantization_error_history = []
        
    def initialize_weights(self, input_dim, random_seed = None):
        if random_seed is not None:
            np.random.seed(random_seed)
        
        num_neurons = self.map_size[0] * self.map_size[1]
        self.weights = np.random.randn(num_neurons, input_dim) * 0.1
        
        self.log(f"Initialized weights: shape {self.weights.shape}")
    
    def create_map_positions(self):
        height, width = self.map_size
        
        if self.topology == 'hexagonal':
            positions = []
            for i in range(height):
                for j in range(width):
                    # Hexagonal offset
                    x = j + (i % 2) * 0.5
                    y = i * np.sqrt(3) / 2
                    positions.append([x, y])
            self.map_positions = np.array(positions)
        else:  # rectangular
            positions = []
            for i in range(height):
                for j in range(width):
                    positions.append([i, j])
            self.map_positions = np.array(positions)
        
        self.log(f"Created {self.topology} map positions: shape {self.map_positions.shape}")
    
    def get_bmu(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=1)
        bmu_idx = np.argmin(distances)
        return bmu_idx
    
    def calculate_neighborhood(self, bmu_idx, epoch):
        progress = epoch / self.max_epochs
        learning_rate = self.learning_rate_initial - (
            self.learning_rate_initial - self.learning_rate_final
        ) * progress
        
        neighborhood_radius = self.neighborhood_initial * np.exp(-progress * 3)
        
        bmu_pos = self.map_positions[bmu_idx]
        distances_on_map = np.linalg.norm(
            self.map_positions - bmu_pos,
            axis=1
        )
        
        neighborhood = np.exp(-(distances_on_map ** 2) / (2 * neighborhood_radius ** 2))
        
        return neighborhood, learning_rate
    
    def update_weights(self, input_vector, bmu_idx, neighborhood, learning_rate):
        update = learning_rate * neighborhood[:, np.newaxis] * (
            input_vector - self.weights
        )
        self.weights += update
    
    def normalize_data(self, data):
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0)
        self.data_std[self.data_std == 0] = 1
        return (data - self.data_mean) / self.data_std
    
    def denormalize_data(self, data):
        if self.data_mean is None or self.data_std is None:
            return data
        return data * self.data_std + self.data_mean
    
    def calculate_quantization_error(self, data):
        errors = []
        for sample in data:
            bmu_idx = self.get_bmu(sample)
            error = np.linalg.norm(sample - self.weights[bmu_idx])
            errors.append(error)
        return np.mean(errors)
    
    def fit(self, problem_data, callback=None, **kwargs):
        self.start_time = time.time()
        
        if 'X' not in problem_data:
            raise ValueError("problem_data must contain 'X' key with input data")
        
        data = problem_data['X']
        if isinstance(data, list):
            data = np.array(data)
        
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D data, got shape {data.shape}")
        
        n_samples, input_dim = data.shape
        self.log(f"Training SOM on {n_samples} samples with {input_dim} features")
        
        random_seed = kwargs.get('random_seed', None)
        
        data_normalized = self.normalize_data(data)
        
        self.initialize_weights(input_dim, random_seed)
        self.create_map_positions()
        
        self.convergence_history = []
        self.bmu_history = []
        self.quantization_error_history = []
        
        for epoch in range(self.max_epochs):
            indices = np.random.permutation(n_samples)
            
            bmu_counts = []
            
            for idx in indices:
                input_vector = data_normalized[idx]
                
                bmu_idx = self.get_bmu(input_vector)
                bmu_counts.append(bmu_idx)
                
                neighborhood, learning_rate = self.calculate_neighborhood(bmu_idx, epoch)
                
                self.update_weights(input_vector, bmu_idx, neighborhood, learning_rate)
            
            quantization_error = self.calculate_quantization_error(data_normalized)
            self.convergence_history.append(quantization_error)
            self.quantization_error_history.append(quantization_error)
            
            unique_bmus = len(np.unique(bmu_counts))
            self.bmu_history.append(unique_bmus)

            if callback:
                callback({
                    'method': 'SOM',
                    'epoch': epoch + 1,
                    'max_epochs': self.max_epochs,
                    'quantization_error': quantization_error,
                    'active_neurons': unique_bmus
                })
            
            if (epoch + 1) % max(1, self.max_epochs // 10) == 0:
                self.log(
                    f"Epoch {epoch + 1}/{self.max_epochs} - "
                    f"Quantization Error: {quantization_error:.6f} - "
                    f"Active Neurons: {unique_bmus}/{self.map_size[0] * self.map_size[1]}"
                )
        
        self.end_time = time.time()
        
        self.results = {
            'weights': self.denormalize_data(self.weights),
            'map_size': self.map_size,
            'topology': self.topology,
            'quantization_error': self.convergence_history[-1],
            'final_active_neurons': self.bmu_history[-1],
            'training_time': self.end_time - self.start_time
        }
        
        self.log(
            f"Training completed in {self.results['training_time']:.2f} seconds. "
            f"Final quantization error: {self.results['quantization_error']:.6f}"
        )
        
        return self.results
    
    def predict(self, data):
        if self.weights is None:
            raise RuntimeError("Model must be trained before prediction")
        
        if isinstance(data, list):
            data = np.array(data)
        
        data_normalized = (data - self.data_mean) / self.data_std
        
        bmu_indices = []
        distances = []
        
        for sample in data_normalized:
            bmu_idx = self.get_bmu(sample)
            distance = np.linalg.norm(sample - self.weights[bmu_idx])
            bmu_indices.append(bmu_idx)
            distances.append(distance)
        
        return np.array(bmu_indices), np.array(distances)
    
    def get_map_grid(self):
        if self.weights is None:
            raise RuntimeError("Model must be trained before getting map grid")
        
        height, width = self.map_size
        n_features = self.weights.shape[1]
        
        grid = self.weights.reshape(height, width, n_features)
        return grid, self.map_size
    
    def get_activation_map(self, data):
        bmu_indices, _ = self.predict(data)
        
        height, width = self.map_size
        activation_map = np.zeros((height * width,))
        
        np.add.at(activation_map, bmu_indices, 1)
        
        return activation_map.reshape(height, width)
