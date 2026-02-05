"""
Pydantic schemas for MetaMind orchestrator.
Ensures structured LLM outputs and type safety.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
import logging


logger = logging.getLogger(__name__)


class LLMRecommendationSchema(BaseModel):
    """
    Schema for LLM recommendations.
    Ensures the LLM output matches expected structure via Pydantic validation.
    """
    
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
        """Ensure confidence is valid probability."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return round(v, 3)
    
    @field_validator('expected_performance')
    @classmethod
    def validate_performance(cls, v: str) -> str:
        """Ensure performance is valid level."""
        valid_levels = {'low', 'medium', 'high'}
        if v.lower() not in valid_levels:
            raise ValueError(f"Performance must be one of {valid_levels}")
        return v.lower()
    
    class Config:
        """Pydantic config."""
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
    """
    Schema for feedback-based parameter adjustments.
    Extends LLMRecommendationSchema with additional feedback context.
    """
    
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
    """Summary of a method execution."""
    
    method_name: str
    problem_name: str
    best_fitness: float
    iterations: int
    execution_time: float
    gap_percentage: Optional[float] = None
    success: bool
    error_message: Optional[str] = None


class OrchestrationSessionSchema(BaseModel):
    """
    Schema for an orchestration session.
    Tracks a complete solve from recommendation to execution.
    """
    
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


class OrchestratorStatsSchema(BaseModel):
    """Statistics about orchestrator usage."""
    
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
