from enum import Enum
from typing import Dict, List, Any, Union, Optional, Callable
from dataclasses import dataclass
import numpy as np


class ProblemType(Enum):
    OPTIMIZATION = "optimization"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    REGRESSION = "regression"


class MethodCategory(Enum):
    NEURAL_NETWORK = "neural_network"
    FUZZY_SYSTEM = "fuzzy_system"
    EVOLUTIONARY = "evolutionary"
    SWARM_INTELLIGENCE = "swarm_intelligence"
    HYBRID = "hybrid"


class OptimizationType(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class MethodInfo:
    name: str
    category: MethodCategory
    applicable_problems: List[ProblemType]
    strengths: List[str]
    weaknesses: List[str]
    typical_parameters: Dict[str, Any]
    complexity: str  # 'low', 'medium', 'high'
    

@dataclass
class ExecutionResult:
    method_name: str
    problem_name: str
    best_solution: Any
    best_fitness: float
    convergence_history: List[float]
    execution_time: float
    iterations: int
    parameters_used: Dict[str, Any]
    metrics: Dict[str, float]
    logs: List[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None
    interpretation: Optional[Dict[str, Any]] = None


@dataclass
class LLMRecommendation:
    selected_method: str
    reasoning: str
    parameters: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    alternative_methods: List[str]
    expected_performance: str
    warnings: List[str]
    backup_strategy: Optional[str] = None


@dataclass
class ProblemDescription:
    problem_type: ProblemType
    description: str
    data_characteristics: Dict[str, Any]
    constraints: List[str]
    objectives: List[str]
    performance_priority: str  # 'speed', 'accuracy', 'balanced'
    max_time: Optional[float] = None
    target_accuracy: Optional[float] = None


# Type aliases for common structures
SolutionType = Union[np.ndarray, List[int], List[float], Any]
FitnessFunction = Callable[[SolutionType], float]
ParameterDict = Dict[str, Union[int, float, str, bool, List, Dict]]
MetricsDict = Dict[str, Union[int, float, str]]
