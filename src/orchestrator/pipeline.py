"""
MetaMind Orchestrator Pipeline - Central orchestration engine.
Coordinates method selection, execution, feedback loops, and result interpretation.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, Type
from importlib import import_module

from src.core.base_method import BaseMethod
from src.core.base_problem import BaseProblem
from src.core.types import ExecutionResult, LLMRecommendation, MethodInfo
from .agent import MetaMindAgent
from .prompts import get_default_method_mapping
from .schema import LLMRecommendationSchema


logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central orchestration engine for MetaMind framework.
    Handles method selection, execution, and feedback loops.
    """
    
    # Method registry mapping method names to their classes
    METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {}
    
    def __init__(
        self,
        groq_api_key: str,
        model: str = "llama-3.3-70b-versatile",
        verbose: bool = True,
        enable_feedback_loop: bool = True,
        max_feedback_iterations: int = 2
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            groq_api_key: Groq API key for LLM
            model: LLM model identifier
            verbose: Enable logging
            enable_feedback_loop: Enable parameter tuning feedback loop
            max_feedback_iterations: Maximum feedback iterations
        """
        self.agent = MetaMindAgent(
            api_key=groq_api_key,
            model=model,
            verbose=verbose
        )
        self.verbose = verbose
        self.enable_feedback_loop = enable_feedback_loop
        self.max_feedback_iterations = max_feedback_iterations
        self.execution_history = []
        
        # Auto-register methods if not already done
        if not self.METHOD_REGISTRY:
            self._register_default_methods()
        
        if verbose:
            logger.info(f"Orchestrator initialized with {len(self.METHOD_REGISTRY)} methods")
    
    @classmethod
    def _register_default_methods(cls):
        """Register default methods from mapping."""
        method_mapping = get_default_method_mapping()
        
        for method_name, import_path in method_mapping.items():
            try:
                module_path, class_name = import_path.rsplit(".", 1)
                module = import_module(module_path)
                method_class = getattr(module, class_name)
                cls.METHOD_REGISTRY[method_name] = method_class
                logger.info(f"Registered method: {method_name}")
            except ImportError as e:
                logger.warning(f"Failed to register {method_name}: {e}")
    
    @classmethod
    def register_method(cls, name: str, method_class: Type[BaseMethod]) -> None:
        """
        Register a custom method.
        
        Args:
            name: Method name
            method_class: Method class (must inherit from BaseMethod)
        """
        if not issubclass(method_class, BaseMethod):
            raise TypeError(f"{method_class} must inherit from BaseMethod")
        cls.METHOD_REGISTRY[name] = method_class
        if logger:
            logger.info(f"Registered custom method: {name}")
    
    def get_available_methods_specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get PARAM_SPECS for all registered methods.
        
        Returns:
            Dictionary mapping method names to their parameter specifications
        """
        specs = {}
        for name, method_class in self.METHOD_REGISTRY.items():
            specs[name] = method_class.PARAM_SPECS
        return specs
    
    def _instantiate_method(
        self,
        method_name: str,
        parameters: Dict[str, Any]
    ) -> BaseMethod:
        """
        Instantiate a method with parameters.
        
        Args:
            method_name: Name of the method
            parameters: Parameter dictionary
            
        Returns:
            Instantiated method instance
            
        Raises:
            KeyError: If method not registered
            ValueError: If parameters are invalid
        """
        if method_name not in self.METHOD_REGISTRY:
            raise KeyError(f"Method '{method_name}' not registered. Available: {list(self.METHOD_REGISTRY.keys())}")
        
        method_class = self.METHOD_REGISTRY[method_name]
        
        try:
            instance = method_class(**parameters)
            if self.verbose:
                logger.info(f"Instantiated {method_name} with parameters: {parameters}")
            return instance
        except Exception as e:
            logger.error(f"Failed to instantiate {method_name}: {e}")
            raise
    
    def _prepare_problem_data(self, problem: BaseProblem) -> Dict[str, Any]:
        """
        Prepare problem data for method execution.
        Different methods expect different data formats.
        
        Args:
            problem: BaseProblem instance
            
        Returns:
            Formatted problem data
        """
        problem_data = {}
        
        # Handle classification/clustering problems with training data
        if hasattr(problem, 'get_train_data'):
            try:
                X_train, y_train = problem.get_train_data()
                problem_data['X_train'] = X_train
                problem_data['y_train'] = y_train
                
                if hasattr(problem, 'get_test_data'):
                    X_test, y_test = problem.get_test_data()
                    problem_data['X_test'] = X_test
                    problem_data['y_test'] = y_test
            except Exception as e:
                logger.warning(f"Could not extract training data: {e}")
        
        # Handle optimization problems
        if hasattr(problem, 'evaluate') and hasattr(problem, 'get_bounds'):
            problem_data['objective_function'] = problem.evaluate
            bounds = problem.get_bounds()
            if bounds is not None:
                problem_data['bounds'] = bounds
        
        # Include problem object itself as fallback
        if not problem_data:
            problem_data = problem
        
        return problem_data
    
    def solve(
        self,
        problem: BaseProblem,
        context: Optional[str] = None
    ) -> Tuple[ExecutionResult, Dict[str, Any]]:
        """
        Solve a problem using the orchestrator pipeline.
        Main entry point for the framework.
        
        Args:
            problem: Problem instance (inherits from BaseProblem)
            context: Optional additional context for LLM
            
        Returns:
            Tuple of (ExecutionResult, LLM recommendation)
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"Starting solve for: {problem.problem_name}")
        
        # Phase 1: Get problem information
        problem_info = problem.get_info()
        available_methods = self.get_available_methods_specs()
        
        # Phase 2: Get LLM recommendation
        recommendation = self.agent.get_recommendation(
            problem_info=problem_info,
            available_methods=available_methods,
            context=context
        )
        
        # Phase 3: Instantiate and execute method
        problem_data = self._prepare_problem_data(problem)
        method_instance = self._instantiate_method(
            recommendation.selected_method,
            recommendation.parameters
        )
        
        # Execute fit
        method_instance.fit(problem_data)
        initial_results = method_instance.get_results()
        initial_metrics = problem.compute_metrics(initial_results.get('best_solution'))
        
        if self.verbose:
            logger.info(
                f"Initial execution complete. "
                f"Best fitness: {initial_metrics.get('fitness', 'N/A')}"
            )
        
        # Phase 4: Feedback loop (optional)
        final_results = initial_results
        final_metrics = initial_metrics
        final_recommendation = recommendation.model_dump()
        
        if self.enable_feedback_loop and 'gap_percentage' in initial_metrics:
            gap = initial_metrics['gap_percentage']
            
            if self.verbose:
                logger.info(f"Gap to optimal: {gap:.2f}%. Starting feedback loop...")
            
            # Iteratively improve if gap is significant
            for iteration in range(1, self.max_feedback_iterations + 1):
                if gap < 5.0:  # Good enough
                    if self.verbose:
                        logger.info(f"Gap below 5%. Stopping feedback loop.")
                    break
                
                # Get feedback recommendation
                feedback_rec = self.agent.get_feedback_recommendation(
                    problem_info=problem_info,
                    available_methods=available_methods,
                    previous_result={
                        'best_fitness': final_metrics.get('fitness'),
                        'metrics': final_metrics
                    },
                    previous_recommendation=final_recommendation
                )
                
                if self.verbose:
                    logger.info(
                        f"Feedback iteration {iteration}: "
                        f"{feedback_rec.selected_method} "
                        f"(confidence: {feedback_rec.confidence:.2f})"
                    )
                
                # Re-execute with adjusted parameters
                problem_data = self._prepare_problem_data(problem)
                method_instance = self._instantiate_method(
                    feedback_rec.selected_method,
                    feedback_rec.parameters
                )
                
                method_instance.fit(problem_data)
                new_results = method_instance.get_results()
                new_metrics = problem.compute_metrics(new_results.get('best_solution'))
                
                # Check for improvement
                if new_metrics.get('fitness', float('inf')) < final_metrics.get('fitness', float('inf')):
                    final_results = new_results
                    final_metrics = new_metrics
                    final_recommendation = feedback_rec.model_dump()
                    gap = new_metrics.get('gap_percentage', gap)
                    
                    if self.verbose:
                        logger.info(f"Improvement found. New gap: {gap:.2f}%")
                else:
                    if self.verbose:
                        logger.info("No improvement. Stopping feedback loop.")
                    break
        
        # Phase 5: Create execution result
        execution_time = time.time() - start_time
        
        execution_result = ExecutionResult(
            method_name=final_recommendation.get('selected_method', 'Unknown'),
            problem_name=problem.problem_name,
            best_solution=final_results.get('best_solution'),
            best_fitness=final_metrics.get('fitness', float('nan')),
            convergence_history=method_instance.convergence_history,
            execution_time=execution_time,
            iterations=len(method_instance.convergence_history),
            parameters_used=final_recommendation.get('parameters', {}),
            metrics=final_metrics,
            logs=method_instance.execution_log,
            success=True,
            error_message=None
        )
        
        # Phase 6: LLM Interpretation and Recommendations
        try:
            if self.verbose:
                logger.info("Starting result interpretation...")
            
            interpretation = self.agent.interpret_results(
                problem_info=problem_info,
                execution_result={
                    'best_fitness': execution_result.best_fitness,
                    'execution_time': execution_result.execution_time,
                    'iterations': execution_result.iterations,
                    'metrics': execution_result.metrics
                },
                recommendation=final_recommendation
            )
            
            execution_result.interpretation = interpretation
            
            if self.verbose:
                logger.info(
                    f"Interpretation complete: "
                    f"{interpretation.get('performance_assessment')} "
                    f"({interpretation.get('confidence_assessment')} confidence)"
                )
        except Exception as e:
            logger.warning(f"Could not generate result interpretation: {e}")
            # Continue without interpretation rather than failing
        
        # Store in history
        self.execution_history.append({
            'problem': problem.problem_name,
            'method': final_recommendation.get('selected_method'),
            'time': execution_time,
            'fitness': final_metrics.get('fitness'),
            'timestamp': time.time()
        })
        
        if self.verbose:
            logger.info(
                f"Solve complete. "
                f"Method: {execution_result.method_name}, "
                f"Fitness: {execution_result.best_fitness:.4f}, "
                f"Time: {execution_time:.2f}s"
            )
        
        return execution_result, final_recommendation
    
    def batch_solve(
        self,
        problems: list,
        contexts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Tuple[ExecutionResult, Dict[str, Any]]]:
        """
        Solve multiple problems.
        
        Args:
            problems: List of problem instances
            contexts: Optional mapping of problem names to contexts
            
        Returns:
            Dictionary mapping problem names to (result, recommendation) tuples
        """
        results = {}
        contexts = contexts or {}
        
        for problem in problems:
            context = contexts.get(problem.problem_name)
            try:
                result, rec = self.solve(problem, context)
                results[problem.problem_name] = (result, rec)
            except Exception as e:
                logger.error(f"Failed to solve {problem.problem_name}: {e}")
                results[problem.problem_name] = (None, None)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_executions': len(self.execution_history),
            'agent_calls': self.agent.call_count,
            'registered_methods': len(self.METHOD_REGISTRY),
            'methods': list(self.METHOD_REGISTRY.keys()),
            'agent_stats': self.agent.get_stats(),
            'execution_history': self.execution_history[-10:]  # Last 10
        }
