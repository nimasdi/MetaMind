from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
import logging


logger = logging.getLogger(__name__)


class LLMRecommendationSchema(BaseModel):

    selected_method: str = Field(
        ...,
        description="Name of the CI method class to use (e.g., 'ACO', 'GA', 'MLP')"
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation for why this method was chosen"
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="Dictionary of method-specific parameters matching PARAM_SPECS"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of this recommendation (0.0 to 1.0)"
    )
    alternative_methods: List[str] = Field(
        default_factory=list,
        description="List of alternative method names that could also work"
    )
    expected_performance: str = Field(
        ...,
        description="Expected performance level: 'low', 'medium', or 'high'"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of potential issues or concerns"
    )
    backup_strategy: Optional[str] = Field(
        None,
        description="Optional backup approach if performance is poor"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return round(v, 3)
    
    @field_validator('expected_performance')
    @classmethod
    def validate_performance(cls, v: str) -> str:
        v_lower = v.lower().strip()
        
        valid_levels = {'low', 'medium', 'high'}
        if v_lower in valid_levels:
            return v_lower
        
        performance_mapping = {
            'medium-high': 'high',
            'medium high': 'high',
            'high-medium': 'high',
            'low-medium': 'medium',
            'medium-low': 'medium',
            'very high': 'high',
            'very low': 'low',
        }
        
        if v_lower in performance_mapping:
            return performance_mapping[v_lower]
        
        raise ValueError(f"Performance must be one of {valid_levels} (got: {v})")
        return v.lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "selected_method": "ACO",
                "reasoning": "TSP problems are best solved with ACO due to pheromone trails",
                "parameters": {
                    "n_ants": 50,
                    "max_iterations": 500,
                    "alpha": 1.0,
                    "beta": 2.0,
                    "evaporation_rate": 0.5
                },
                "confidence": 0.95,
                "alternative_methods": ["PSO", "GA"],
                "expected_performance": "high",
                "warnings": ["Requires good parameter tuning"],
                "backup_strategy": "Try PSO if performance is poor"
            }
        }


class FeedbackRecommendationSchema(LLMRecommendationSchema):
    adjustment_reason: str = Field(
        ...,
        description="Explanation of why these parameters were adjusted"
    )
    expected_improvement: str = Field(
        ...,
        description="Expected improvement or change from adjustments"
    )
    changed_parameters: Dict[str, tuple] = Field(
        default_factory=dict,
        description="Mapping of changed parameter names to (old_value, new_value) tuples"
    )


class ExecutionSummarySchema(BaseModel):
    method_name: str
    problem_name: str
    best_fitness: float
    iterations: int
    execution_time: float
    gap_percentage: Optional[float] = None
    success: bool
    error_message: Optional[str] = None


class OrchestrationSessionSchema(BaseModel):
    session_id: str
    problem_name: str
    problem_type: str
    initial_recommendation: LLMRecommendationSchema
    execution_summary: ExecutionSummarySchema
    feedback_iterations: List[FeedbackRecommendationSchema] = Field(
        default_factory=list,
        description="List of feedback iterations applied"
    )
    total_time: float
    timestamp: float


class MultiMethodRecommendationSchema(BaseModel):    
    selected_methods: List[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of method names to run (2-5 methods)"
    )
    reasoning: str = Field(
        ...,
        description="Explanation for why these methods were chosen"
    )
    method_parameters: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Mapping of method names to their parameter dictionaries"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of this multi-method recommendation"
    )
    comparison_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria to use when comparing results (e.g., 'fitness', 'execution_time')"
    )
    expected_best_method: Optional[str] = Field(
        None,
        description="Method expected to perform best (if predictable)"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return round(v, 3)
    
    class Config:
        json_schema_extra = {
            "example": {
                "selected_methods": ["ACO", "PSO", "GA"],
                "reasoning": "TSP can be solved by multiple metaheuristics; comparing ACO, PSO, and GA",
                "method_parameters": {
                    "ACO": {"n_ants": 50, "max_iterations": 500, "alpha": 1.0, "beta": 2.0, "evaporation_rate": 0.5},
                    "PSO": {"n_particles": 50, "max_iterations": 500, "w": 0.7, "c1": 1.5, "c2": 1.5},
                    "GA": {"population_size": 50, "max_generations": 500, "crossover_rate": 0.8, "mutation_rate": 0.1}
                },
                "confidence": 0.9,
                "comparison_criteria": ["best_fitness", "execution_time", "convergence_speed"],
                "expected_best_method": "ACO"
            }
        }


class MultiMethodResultAnalysisSchema(BaseModel):    
    recommended_method: str = Field(
        ...,
        description="The method that performed best overall"
    )
    ranking: List[str] = Field(
        ...,
        description="Methods ranked from best to worst"
    )
    analysis: str = Field(
        ...,
        description="Detailed analysis of why the recommended method is best"
    )
    performance_comparison: Dict[str, str] = Field(
        default_factory=dict,
        description="Brief performance summary for each method"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the recommendation"
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended next steps or improvements"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return round(v, 3)


class OrchestratorStatsSchema(BaseModel):
    total_problems_solved: int
    total_executions: int
    agent_calls: int
    registered_methods: int
    method_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of method names to usage counts"
    )
    average_execution_time: float
    average_fitness: float
    success_rate: float
