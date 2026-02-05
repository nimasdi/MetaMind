"""
MetaMind Orchestrator Package

A production-ready orchestration system for Computational Intelligence methods.
Implements an agentic pattern using Groq's LLM for intelligent method selection
and parameter configuration with feedback loops.

Core Components:
- Agent (agent.py): LLM-based method selector using Groq
- Pipeline (pipeline.py): Orchestration engine with feedback loops
- Prompts (prompts.py): Intelligent prompt builders for LLM
- Schema (schema.py): Pydantic models for structured outputs

Usage:
    from src.orchestrator import Orchestrator
    
    # Initialize orchestrator
    orch = Orchestrator(groq_api_key="your_key")
    
    # Solve a problem
    result, recommendation = orch.solve(problem_instance)
    
    # Get statistics
    stats = orch.get_stats()

Architecture:
    1. Problem Analysis: Extract metadata via problem.get_info()
    2. Method Selection: LLM recommends best method based on problem characteristics
    3. Parameter Tuning: LLM configures method parameters within PARAM_SPECS ranges
    4. Execution: Run selected method with optimized parameters
    5. Feedback Loop: Analyze results, suggest parameter adjustments
    6. Iteration: Repeat execution with improved parameters if needed

All 9 CI Methods Supported:
    Neural Networks:
        - Perceptron: Single-layer linear classifier
        - MLP: Multi-layer feedforward network
        - Hopfield: Recurrent associative memory
        - SOM: Self-organizing map for clustering
    
    Fuzzy Systems:
        - FuzzyController: Fuzzy logic inference system
    
    Evolutionary:
        - GA: Genetic algorithm
        - GP: Genetic programming
        - PSO: Particle swarm optimization
        - ACO: Ant colony optimization
"""

from .agent import MetaMindAgent
from .pipeline import Orchestrator
from .prompts import PromptBuilder, get_default_method_mapping
from .schema import (
    LLMRecommendationSchema,
    FeedbackRecommendationSchema,
    ExecutionSummarySchema,
    OrchestrationSessionSchema,
    OrchestratorStatsSchema,
)

__all__ = [
    "Orchestrator",
    "MetaMindAgent",
    "PromptBuilder",
    "LLMRecommendationSchema",
    "FeedbackRecommendationSchema",
    "ExecutionSummarySchema",
    "OrchestrationSessionSchema",
    "OrchestratorStatsSchema",
    "get_default_method_mapping",
]

__version__ = "1.0.0"
__author__ = "MetaMind Team"
