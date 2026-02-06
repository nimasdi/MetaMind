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
    METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {}
    
    def __init__(
        self,
        api_key: str,
        model: str = "NousResearch/Hermes-3-Llama-3.1-70B",
        verbose: bool = True,
        enable_feedback_loop: bool = True,
        max_feedback_iterations: int = 2
    ):

        self.agent = MetaMindAgent(
            api_key=api_key,
            model=model,
            verbose=verbose
        )
        self.verbose = verbose
        self.enable_feedback_loop = enable_feedback_loop
        self.max_feedback_iterations = max_feedback_iterations
        self.execution_history = []
        
        if not self.METHOD_REGISTRY:
            self._register_default_methods()
        
        if verbose:
            logger.info(f"Orchestrator initialized with {len(self.METHOD_REGISTRY)} methods")
    
    @classmethod
    def _register_default_methods(cls):
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
        if not issubclass(method_class, BaseMethod):
            raise TypeError(f"{method_class} must inherit from BaseMethod")
        cls.METHOD_REGISTRY[name] = method_class
        if logger:
            logger.info(f"Registered custom method: {name}")
    
    def get_available_methods_specs(self) -> Dict[str, Dict[str, Any]]:
        specs = {}
        for name, method_class in self.METHOD_REGISTRY.items():
            specs[name] = method_class.PARAM_SPECS
        return specs
    
    def _instantiate_method(
        self,
        method_name: str,
        parameters: Dict[str, Any]
    ) -> BaseMethod:
        if method_name not in self.METHOD_REGISTRY:
            raise KeyError(f"Method '{method_name}' not registered. Available: {list(self.METHOD_REGISTRY.keys())}")
        
        method_class = self.METHOD_REGISTRY[method_name]
        
        full_parameters = method_class.get_default_parameters()
        full_parameters.update(parameters)
        
        try:
            instance = method_class(**full_parameters)
            if self.verbose:
                logger.info(f"Instantiated {method_name} with parameters: {full_parameters}")
            return instance
        except Exception as e:
            logger.error(f"Failed to instantiate {method_name}: {e}")
            raise
    
    def _prepare_problem_data(self, problem: BaseProblem):
        if self.verbose:
            problem_type = problem.problem_type if hasattr(problem, 'problem_type') else 'unknown'
            logger.info(f"Prepared problem data for {problem.problem_name} (type: {problem_type})")
        
        return problem
    
    def _create_progress_callback(self, verbose=True):
        def progress_reporter(metrics: Dict):
            if not verbose:
                return
            
            method = metrics.get('method', 'Unknown')
            step = metrics.get('iteration') or metrics.get('epoch') 
            total = metrics.get('max_iterations') or metrics.get('max_epochs')


            status_parts = [f"[{method}] Step {step}/{total}"]

            if 'best_fitness' in metrics:
                status_parts.append(f"Fitness: {metrics['best_fitness']:.4f}")
            if 'val_accuracy' in metrics:
                status_parts.append(f"Val Acc: {metrics['val_accuracy']:.4f}")
            if 'val_loss' in metrics:
                status_parts.append(f"Loss: {metrics['val_loss']:.4f}")

            print(" |".join(status_parts), end='\r', flush=True)

        return progress_reporter
    
    def solve(
        self,
        problem: BaseProblem,
        context: Optional[str] = None
    ) -> Tuple[ExecutionResult, Dict[str, Any]]:
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"Starting solve for: {problem.problem_name}")
        

        problem_info = problem.get_info()
        available_methods = self.get_available_methods_specs()

        recommendation = self.agent.get_recommendation(
            problem_info=problem_info,
            available_methods=available_methods,
            context=context
        )

        problem_data = self._prepare_problem_data(problem)
        method_instance = self._instantiate_method(
            recommendation.selected_method,
            recommendation.parameters
        )

        my_callback = self._create_progress_callback(verbose=self.verbose)

        method_instance.fit(problem_data, callback=my_callback)
        initial_results = method_instance.get_results()
        initial_metrics = problem.compute_metrics(initial_results.get('best_solution'))
        
        if self.verbose:
            logger.info(
                f"Initial execution complete. "
                f"Best fitness: {initial_metrics.get('fitness', 'N/A')}"
            )

        final_results = initial_results
        final_metrics = initial_metrics
        final_recommendation = recommendation.model_dump()
        
        if self.enable_feedback_loop and 'gap_percentage' in initial_metrics:
            gap = initial_metrics['gap_percentage']
            
            if self.verbose:
                logger.info(f"Gap to optimal: {gap:.2f}%. Starting feedback loop...")
            
            for iteration in range(1, self.max_feedback_iterations + 1):
                if gap < 5.0:
                    if self.verbose:
                        logger.info(f"Gap below 5%. Stopping feedback loop.")
                    break
                
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
                
                problem_data = self._prepare_problem_data(problem)
                method_instance = self._instantiate_method(
                    feedback_rec.selected_method,
                    feedback_rec.parameters
                )
                
                method_instance.fit(problem_data)
                new_results = method_instance.get_results()
                new_metrics = problem.compute_metrics(new_results.get('best_solution'))
                
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
    
    def multi_solve(
        self,
        problem: BaseProblem,
        num_methods: int = 3,
        context: Optional[str] = None
    ) -> Tuple[ExecutionResult, Dict[str, Any], Dict[str, Any]]:
        start_time = time.time()
        
        if self.verbose:
            logger.info(
                f"Starting multi_solve with {num_methods} methods for: "
                f"{problem.problem_name}"
            )
        
        problem_info = problem.get_info()
        available_methods = self.get_available_methods_specs()

        multi_recommendation = self.agent.get_multi_method_recommendation(
            problem_info=problem_info,
            available_methods=available_methods,
            num_methods=num_methods,
            context=context
        )
        
        if self.verbose:
            logger.info(
                f"Selected methods: {', '.join(multi_recommendation.selected_methods)}"
            )

        all_results = {}
        all_execution_results = {}
        
        for method_name in multi_recommendation.selected_methods:
            if self.verbose:
                logger.info(f"Executing {method_name}...")
            
            try:
                method_start = time.time()

                parameters = multi_recommendation.method_parameters.get(method_name, {})
                
                problem_data = self._prepare_problem_data(problem)
                method_instance = self._instantiate_method(method_name, parameters)
                
                my_callback = self._create_progress_callback(verbose=self.verbose)
                method_instance.fit(problem_data, callback=my_callback)
                
                results = method_instance.get_results()
                metrics = problem.compute_metrics(results.get('best_solution'))
                
                method_time = time.time() - method_start

                all_results[method_name] = {
                    'best_fitness': metrics.get('fitness', float('nan')),
                    'execution_time': method_time,
                    'iterations': len(method_instance.convergence_history),
                    'metrics': metrics,
                    'success': True,
                    'convergence_history': method_instance.convergence_history
                }

                all_execution_results[method_name] = ExecutionResult(
                    method_name=method_name,
                    problem_name=problem.problem_name,
                    best_solution=results.get('best_solution'),
                    best_fitness=metrics.get('fitness', float('nan')),
                    convergence_history=method_instance.convergence_history,
                    execution_time=method_time,
                    iterations=len(method_instance.convergence_history),
                    parameters_used=parameters,
                    metrics=metrics,
                    logs=method_instance.execution_log,
                    success=True,
                    error_message=None
                )
                
                if self.verbose:
                    logger.info(
                        f"{method_name} completed: fitness={metrics.get('fitness', 'N/A'):.4f}, "
                        f"time={method_time:.2f}s"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to execute {method_name}: {e}")
                all_results[method_name] = {
                    'best_fitness': float('inf'),
                    'execution_time': 0.0,
                    'iterations': 0,
                    'metrics': {},
                    'success': False,
                    'error': str(e)
                }
                all_execution_results[method_name] = None

        if self.verbose:
            logger.info("Analyzing results with LLM...")
        
        analysis = self.agent.analyze_multi_method_results(
            problem_info=problem_info,
            execution_results=all_results
        )

        best_method = analysis.recommended_method
        best_execution_result = all_execution_results.get(best_method)
        
        if best_execution_result is None:
            successful_methods = {
                m: r for m, r in all_execution_results.items() if r is not None
            }
            
            if not successful_methods:
                error_msg = "All methods failed to execute. Check problem data format and method requirements."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            best_method = min(
                successful_methods.keys(),
                key=lambda m: all_results[m].get('best_fitness', float('inf'))
            )
            best_execution_result = successful_methods[best_method]
            logger.warning(
                f"LLM recommended method {analysis.recommended_method} not available, "
                f"using {best_method} instead"
            )
        
        best_execution_result.interpretation = {
            'multi_method_analysis': analysis.model_dump(),
            'all_method_results': {
                method: {
                    'fitness': res.get('best_fitness'),
                    'time': res.get('execution_time'),
                    'gap': res.get('metrics', {}).get('gap_percentage')
                }
                for method, res in all_results.items()
            }
        }
        
        total_time = time.time() - start_time
 
        self.execution_history.append({
            'problem': problem.problem_name,
            'mode': 'multi_method',
            'methods': multi_recommendation.selected_methods,
            'best_method': best_method,
            'time': total_time,
            'fitness': best_execution_result.best_fitness,
            'timestamp': time.time()
        })
        
        if self.verbose:
            logger.info(
                f"Multi-solve complete. Best method: {best_method}, "
                f"Fitness: {best_execution_result.best_fitness:.4f}, "
                f"Total time: {total_time:.2f}s"
            )
        
        return best_execution_result, analysis.model_dump(), all_results
    
    def batch_solve(
        self,
        problems: list,
        contexts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Tuple[ExecutionResult, Dict[str, Any]]]:
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
        return {
            'total_executions': len(self.execution_history),
            'agent_calls': self.agent.call_count,
            'registered_methods': len(self.METHOD_REGISTRY),
            'methods': list(self.METHOD_REGISTRY.keys()),
            'agent_stats': self.agent.get_stats(),
            'execution_history': self.execution_history[-10:]  
        }




