import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

class BaseMethod(ABC):

    PARAM_SPECS = {}
    
    def __init__(self, **parameters):
        self.parameters = parameters
        self.results = {}
        self.execution_log = []
        self.convergence_history = []
        self.start_time = None
        self.end_time = None
        self._validate_parameters()
        
    def _validate_parameters(self):
        for param_name, value in self.parameters.items():
            if param_name not in self.PARAM_SPECS:
                self.log(f"Warning: Unknown parameter '{param_name}'")
                continue
                
            spec = self.PARAM_SPECS[param_name]
            
            if 'type' in spec and not isinstance(value, spec['type']):
                if spec['type'] == int and isinstance(value, float) and value.is_integer():
                    self.parameters[param_name] = int(value)
                else:
                    raise ValueError(
                        f"Parameter '{param_name}' must be of type {spec['type'].__name__}, "
                        f"got {type(value).__name__}"
                    )
            
            if 'range' in spec:
                min_val, max_val = spec['range']
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Parameter '{param_name}' must be in range [{min_val}, {max_val}], "
                        f"got {value}"
                    )
            
            if 'options' in spec:
                if value not in spec['options']:
                    raise ValueError(
                        f"Parameter '{param_name}' must be one of {spec['options']}, "
                        f"got {value}"
                    )
    
    @abstractmethod
    def fit(self, problem_data, callback: Optional[Callable[[Dict], None]] = None, **kwargs):
        pass
    
    def log(self, message: str):
        log_entry = {
            'timestamp': time.time(),
            'elapsed': time.time() - self.start_time if self.start_time else 0,
            'message': message
        }
        self.execution_log.append(log_entry)
        
    def get_progress(self):
        progress = {
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'convergence_history': self.convergence_history.copy(),
        }
        
        if self.convergence_history:
            progress['current_best'] = self.convergence_history[-1]
            progress['iterations'] = len(self.convergence_history)
            
        return progress
    
    def get_results(self):
        return self.results.copy()
    
    def get_logs(self):
        return self.execution_log.copy()


    
    @classmethod
    def get_default_parameters(cls):
        defaults = {}
        for param_name, spec in cls.PARAM_SPECS.items():
            if 'default' in spec:
                defaults[param_name] = spec['default']
        return defaults

