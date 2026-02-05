"""
Complete example demonstrating MetaMind Orchestrator usage.
Shows how to initialize, configure, and use the orchestration system.
"""

import os
import logging
from typing import Optional

api_key = os.getenv("GROQ_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """
    Basic example: Initialize orchestrator and solve a single problem.
    """
    from src.orchestrator import Orchestrator
    from src.problems.continuous import SphereFunction
    
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Basic Usage - Single Problem")
    logger.info("=" * 60)
    
    # Initialize orchestrator with Groq API key

    orchestrator = Orchestrator(
        groq_api_key=api_key,
        verbose=True,
        enable_feedback_loop=True,
        max_feedback_iterations=2
    )
    
    # Create a problem instance
    problem = SphereFunction()
    problem.load_data()
    
    # Solve using orchestrator
    execution_result, recommendation = orchestrator.solve(
        problem=problem,
        context="Optimize a 10D sphere function for accuracy"
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("EXECUTION RESULTS")
    print("=" * 60)
    print(f"Method: {execution_result.method_name}")
    print(f"Best Fitness: {execution_result.best_fitness:.6f}")
    print(f"Iterations: {execution_result.iterations}")
    print(f"Execution Time: {execution_result.execution_time:.2f}s")
    print(f"Parameters: {execution_result.parameters_used}")
    print(f"Metrics: {execution_result.metrics}")
    
    print("\n" + "=" * 60)
    print("LLM RECOMMENDATION")
    print("=" * 60)
    print(f"Selected Method: {recommendation.get('selected_method')}")
    print(f"Reasoning: {recommendation.get('reasoning')}")
    print(f"Confidence: {recommendation.get('confidence'):.2%}")
    print(f"Expected Performance: {recommendation.get('expected_performance')}")
    print(f"Warnings: {recommendation.get('warnings')}")
    
    return orchestrator


def example_feedback_loop():
    """
    Example showing the feedback loop in action.
    Demonstrates parameter tuning for improved results.
    """
    from src.orchestrator import Orchestrator
    from src.problems.tsp import TSP
    
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Feedback Loop - Parameter Tuning")
    logger.info("=" * 60)
    
    orchestrator = Orchestrator(
        groq_api_key=api_key,
        verbose=True,
        enable_feedback_loop=True,  # Enable feedback loop
        max_feedback_iterations=3    # Allow up to 3 iterations
    )
    
    # Load TSP problem
    tsp = TSP(problem_file="data/tsplib/berlin52.tsp")
    tsp.load_data()
    
    # Solve with automatic parameter tuning
    execution_result, recommendation = orchestrator.solve(tsp)
    
    print(f"\nFinal gap to optimal: {execution_result.metrics.get('gap_percentage', 'N/A'):.2f}%")
    print(f"Total execution time: {execution_result.execution_time:.2f}s")
    
    return orchestrator


def example_batch_processing():
    """
    Example: Solve multiple problems in batch.
    """
    from src.orchestrator import Orchestrator
    from src.problems.continuous import SphereFunction, RastriginFunction
    from src.problems.classification import IrisClassification
    
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Batch Processing - Multiple Problems")
    logger.info("=" * 60)

    orchestrator = Orchestrator(
        groq_api_key=api_key,
        verbose=True,
        enable_feedback_loop=False  # Disable for batch speed
    )
    
    # Create multiple problems
    problems = [
        SphereFunction(),
        RastriginFunction(),
        IrisClassification()
    ]
    
    # Load all problems
    for prob in problems:
        prob.load_data()
    
    # Solve all in batch
    results = orchestrator.batch_solve(
        problems=problems,
        contexts={
            "Sphere10": "10D sphere optimization",
            "Rastrigin10": "Multimodal optimization",
            "Iris": "Classification with Iris dataset"
        }
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("BATCH RESULTS SUMMARY")
    print("=" * 60)
    for problem_name, (result, rec) in results.items():
        if result:
            print(f"\n{problem_name}:")
            print(f"  Method: {result.method_name}")
            print(f"  Fitness: {result.best_fitness:.6f}")
            print(f"  Time: {result.execution_time:.2f}s")
    
    return orchestrator


def example_custom_method_registration():
    """
    Example: Register custom CI methods with orchestrator.
    """
    from src.orchestrator import Orchestrator
    from src.methods.evolutionary.aco import AntColonyOptimization
    
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Custom Method Registration")
    logger.info("=" * 60)
    
    orchestrator = Orchestrator(groq_api_key=api_key)
    
    # Register a custom method (or override existing)
    # Example: Create a custom variant of ACO
    class CustomACO(AntColonyOptimization):
        PARAM_SPECS = AntColonyOptimization.PARAM_SPECS.copy()
        # Could add custom parameters here
    
    # Register it
    Orchestrator.register_method("CustomACO", CustomACO)
    
    # Get available methods
    available = orchestrator.get_available_methods_specs()
    print(f"\nRegistered methods: {list(available.keys())}")
    print(f"CustomACO registered: {'CustomACO' in available}")
    
    return orchestrator


def example_statistics_and_monitoring():
    """
    Example: Monitor orchestrator usage and statistics.
    """
    from src.orchestrator import Orchestrator
    from src.problems.continuous import SphereFunction
    
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Statistics & Monitoring")
    logger.info("=" * 60)
    
    orchestrator = Orchestrator(groq_api_key=api_key, verbose=False)
    
    # Solve a problem
    problem = SphereFunction()
    problem.load_data()
    orchestrator.solve(problem)
    
    # Get statistics
    stats = orchestrator.get_stats()
    
    print("\n" + "=" * 60)
    print("ORCHESTRATOR STATISTICS")
    print("=" * 60)
    print(f"Total Executions: {stats['total_executions']}")
    print(f"Agent Calls: {stats['agent_calls']}")
    print(f"Registered Methods: {stats['registered_methods']}")
    print(f"Available Methods: {stats['methods']}")
    print(f"\nAgent Stats:")
    print(f"  Model: {stats['agent_stats']['model']}")
    print(f"  Temperature: {stats['agent_stats']['temperature']}")
    
    return orchestrator


def example_parameter_specs_exploration():
    """
    Example: Explore parameter specifications for all methods.
    """
    from src.orchestrator import Orchestrator, PromptBuilder
    
    logger.info("=" * 60)
    logger.info("EXAMPLE 6: Parameter Specifications")
    logger.info("=" * 60)
    
    orchestrator = Orchestrator(groq_api_key=api_key)
    
    # Get all parameter specs
    specs = orchestrator.get_available_methods_specs()
    
    print("\n" + "=" * 60)
    print("PARAMETER SPECIFICATIONS BY METHOD")
    print("=" * 60)
    
    for method_name, param_specs in specs.items():
        print(f"\n{method_name}:")
        for param_name, spec in param_specs.items():
            print(f"  {param_name}:")
            if 'range' in spec:
                print(f"    Range: {spec['range']}")
            if 'options' in spec:
                print(f"    Options: {spec['options']}")
            if 'default' in spec:
                print(f"    Default: {spec['default']}")
            if 'type' in spec:
                type_name = spec['type'].__name__ if hasattr(spec['type'], '__name__') else str(spec['type'])
                print(f"    Type: {type_name}")
    
    return orchestrator


if __name__ == "__main__":
    """
    Run all examples.
    Uncomment the examples you want to run.
    """
    
    print("\n" + "=" * 60)
    print("METAMIND ORCHESTRATOR - COMPREHENSIVE EXAMPLES")
    print("=" * 60)
    
    try:
        # Run examples (comment out as needed)
        # example_basic_usage()
        # example_feedback_loop()
        # example_batch_processing()
        # example_custom_method_registration()
        # example_statistics_and_monitoring()
        example_parameter_specs_exploration()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
