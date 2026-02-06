import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from src.orchestrator.pipeline import Orchestrator
from src.problems.tsp import TSPProblem, load_tsplib_instance
from src.problems.continuous import create_benchmark_function


def example_tsp_multi_method():
    print("=" * 60)
    print("MULTI-METHOD ORCHESTRATION EXAMPLE: TSP")
    print("=" * 60)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Please set GROQ_API_KEY environment variable")
        return
    
    orchestrator = Orchestrator(
        api_key=api_key,
        verbose=True,
        enable_feedback_loop=False
    )
    
    tsplib_dir = project_root / "data" / "tsplib"

    problem = load_tsplib_instance('eil51', str(tsplib_dir))
    
    print(f"\nProblem: {problem.problem_name}")
    print(f"Type: {problem.problem_type}")
    print(f"Cities: {problem.n_cities}")
    print(f"Optimal value: {problem.optimal_value}")
    
    print("\n" + "=" * 60)
    print("Running multi_solve with 3 methods...")
    print("=" * 60)
    
    best_result, analysis, all_results = orchestrator.multi_solve(
        problem=problem,
        num_methods=3,
        context="Focus on achieving best solution quality"
    )
    
    print("\n" + "=" * 60)
    print("MULTI-METHOD RESULTS")
    print("=" * 60)
    
    print("\nAll Method Results:")
    for method, result in all_results.items():
        success_marker = "✓" if result.get('success', False) else "✗"
        fitness = result.get('best_fitness', float('inf'))
        time = result.get('execution_time', 0.0)
        gap = result.get('metrics', {}).get('gap_percentage', 'N/A')
        
        print(f"  {success_marker} {method:20s} | Fitness: {fitness:10.2f} | "
              f"Time: {time:6.2f}s | Gap: {gap}%")
    
    print("\n" + "-" * 60)
    print("LLM ANALYSIS & RECOMMENDATION")
    print("-" * 60)
    
    print(f"\nRecommended Method: {analysis['recommended_method']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"\nRanking: {' > '.join(analysis['ranking'])}")
    
    print(f"\nAnalysis:\n{analysis['analysis']}")
    
    if analysis.get('next_steps'):
        print("\nNext Steps:")
        for i, step in enumerate(analysis['next_steps'], 1):
            print(f"  {i}. {step}")
    
    print("\n" + "=" * 60)
    print(f"BEST RESULT: {best_result.method_name}")
    print("=" * 60)
    print(f"Best Fitness: {best_result.best_fitness:.4f}")
    print(f"Gap from Optimal: {best_result.metrics.get('gap_percentage', 'N/A')}%")
    print(f"Execution Time: {best_result.execution_time:.2f}s")
    print(f"Iterations: {best_result.iterations}")


def example_continuous_optimization_multi_method():
    print("\n\n" + "=" * 60)
    print("MULTI-METHOD ORCHESTRATION EXAMPLE: Rastrigin Function")
    print("=" * 60)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Please set GROQ_API_KEY environment variable")
        return
    
    orchestrator = Orchestrator(
        api_key=api_key,
        verbose=True,
        enable_feedback_loop=False
    )
    
    problem = create_benchmark_function('rastrigin', dimension=10)
    
    print(f"\nProblem: {problem.problem_name}")
    print(f"Type: {problem.problem_type}")
    print(f"Dimension: {problem.dimension}")
    print(f"Optimal value: {problem.optimal_value}")
    
    print("\n" + "=" * 60)
    print("Running multi_solve with 2 methods...")
    print("=" * 60)
    
    best_result, analysis, all_results = orchestrator.multi_solve(
        problem=problem,
        num_methods=2,
        context="Rastrigin has many local optima - need good exploration"
    )
    
    print("\n" + "=" * 60)
    print("MULTI-METHOD RESULTS")
    print("=" * 60)
    
    print("\nAll Method Results:")
    for method, result in all_results.items():
        success_marker = "✓" if result.get('success', False) else "✗"
        fitness = result.get('best_fitness', float('inf'))
        time = result.get('execution_time', 0.0)
        gap = result.get('metrics', {}).get('gap_percentage', 'N/A')
        
        print(f"  {success_marker} {method:20s} | Fitness: {fitness:10.4f} | "
              f"Time: {time:6.2f}s | Gap: {gap}%")
    
    print("\n" + "-" * 60)
    print("LLM ANALYSIS & RECOMMENDATION")
    print("-" * 60)
    
    print(f"\nRecommended Method: {analysis['recommended_method']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"\nRanking: {' > '.join(analysis['ranking'])}")
    
    print(f"\nAnalysis:\n{analysis['analysis']}")
    
    if analysis.get('performance_comparison'):
        print("\nPerformance Comparison:")
        for method, summary in analysis['performance_comparison'].items():
            print(f"  • {method}: {summary}")
    
    print("\n" + "=" * 60)
    print(f"BEST RESULT: {best_result.method_name}")
    print("=" * 60)
    print(f"Best Fitness: {best_result.best_fitness:.6f}")
    print(f"Gap from Optimal: {best_result.metrics.get('gap_percentage', 'N/A')}%")
    print(f"Execution Time: {best_result.execution_time:.2f}s")


def save_results_example():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return
    
    orchestrator = Orchestrator(api_key=api_key, verbose=False)

    tsplib_dir = project_root / "data" / "tsplib"

    problem = load_tsplib_instance('berlin52', str(tsplib_dir))
    
    best_result, analysis, all_results = orchestrator.multi_solve(
        problem=problem,
        num_methods=2
    )
    
    output = {
        'problem': problem.problem_name,
        'recommended_method': analysis['recommended_method'],
        'ranking': analysis['ranking'],
        'analysis': analysis['analysis'],
        'confidence': analysis['confidence'],
        'all_results': {
            method: {
                'fitness': res.get('best_fitness'),
                'time': res.get('execution_time'),
                'gap_percentage': res.get('metrics', {}).get('gap_percentage')
            }
            for method, res in all_results.items()
        },
        'best_fitness': best_result.best_fitness,
        'best_gap_percentage': best_result.metrics.get('gap_percentage')
    }
    
    output_file = 'multi_method_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    example_tsp_multi_method()
    
    example_continuous_optimization_multi_method()
    
    save_results_example()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
