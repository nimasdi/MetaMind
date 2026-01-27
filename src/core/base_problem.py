from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseProblem(ABC):
    def __init__(self, problem_name: str = "Unknown Problem"):
        self.problem_name = problem_name
        self.problem_type = None  # 'optimization', 'classification', 'clustering'
        self.data = None
        self.metadata = {}
        self.best_known_solution = None
        self.optimal_value = None
        
    @abstractmethod
    def load_data(self, **kwargs) -> None:
        pass
    
    @abstractmethod
    def evaluate(self, solution: Any) -> float:
        pass
    
    def validate_solution(self, solution: Any) -> bool:
        return True  # Default: no constraints
    
    def get_bounds(self) -> Optional[tuple]:
        return None
    
    def get_dimension(self) -> Optional[int]:
        return None
    
    def compute_metrics(self, solution: Any, additional_data: Optional[Dict] = None) -> Dict[str, float]:
        metrics = {
            'fitness': self.evaluate(solution),
            'is_valid': self.validate_solution(solution)
        }
        
        if self.optimal_value is not None:
            metrics['gap_to_optimal'] = abs(metrics['fitness'] - self.optimal_value)
            metrics['gap_percentage'] = (metrics['gap_to_optimal'] / abs(self.optimal_value)) * 100 if self.optimal_value != 0 else 0
            
        return metrics
    
    def get_info(self) -> Dict[str, Any]:
        info = {
            'name': self.problem_name,
            'type': self.problem_type,
            'metadata': self.metadata.copy(),
        }
        
        dimension = self.get_dimension()
        if dimension is not None:
            info['dimension'] = dimension
            
        bounds = self.get_bounds()
        if bounds is not None:
            info['bounds'] = {
                'lower': bounds[0].tolist() if isinstance(bounds[0], np.ndarray) else bounds[0],
                'upper': bounds[1].tolist() if isinstance(bounds[1], np.ndarray) else bounds[1]
            }
            
        if self.optimal_value is not None:
            info['optimal_value'] = self.optimal_value
            
        return info
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.problem_name}', type='{self.problem_type}')"
