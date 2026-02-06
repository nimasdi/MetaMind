import sys
from pathlib import Path
import numpy as np
import time
import json
import os
from datetime import datetime
from dotenv import load_dotenv


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from src.problems.continuous import (
    create_benchmark_function
)

from src.methods.evolutionary.pso import PSO
from src.methods.evolutionary.ga import GeneticAlgorithm

from src.orchestrator.agent import MetaMindAgent
from src.utils.logging import setup_logger, get_experiment_logger, standard_progress_callback
from src.utils.metrics import compute_statistics, compute_gap_percentage, pairwise_wilcoxon_comparison, print_wilcoxon_summary
from src.utils.plotting import plot_convergence, plot_multiple_convergence, plot_box_comparison, plot_convergence_with_bands

from src.orchestrator.memory import MemoryManager


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def get_llm_recommendation(agent, problem, memory_manager):
    """Get LLM recommendation for function optimization problem."""
    print(f"\n{'='*80}")
    print(f"Asking LLM for recommendation on {problem.problem_name}...")
    print(f"{'='*80}")

    problem_info = problem.get_info()
    
    available_methods = {
        'PSO': PSO.PARAM_SPECS,
        'GeneticAlgorithm': GeneticAlgorithm.PARAM_SPECS,
    }

    memory_str = memory_manager.get_context_string(
        problem_type="continuous_optimization", 
        problem_name=problem.problem_name
    )
    
    try:
        recommendation = agent.get_recommendation(
            problem_info=problem_info,
            available_methods=available_methods,
            context=f"This is a continuous function optimization problem ({problem.function_name}). "
                    f"Dimension: {problem.dimension}. "
                    f"We need to find the global minimum efficiently."
                    f"{memory_str}"
        )
        
        print(f"\nLLM Recommendation:")
        print(f"  Method: {recommendation.selected_method}")
        print(f"  Confidence: {recommendation.confidence:.2%}")
        print(f"  Expected Performance: {recommendation.expected_performance}")
        print(f"  Reasoning: {recommendation.reasoning}")
        print(f"\n  Recommended Parameters:")
        for param, value in recommendation.parameters.items():
            print(f"    - {param}: {value}")
        
        if recommendation.alternative_methods:
            print(f"\n  Alternative Methods: {', '.join(recommendation.alternative_methods)}")
        
        if recommendation.warnings:
            print(f"\n  Warnings:")
            for warning in recommendation.warnings:
                print(f"    - {warning}")
        
        return recommendation
        
    except Exception as e:
        print(f"Error getting LLM recommendation: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_method_from_recommendation(recommendation):
    """Instantiate method from LLM recommendation."""
    method_map = {
        'PSO': PSO,
        'ParticleSwarmOptimization': PSO,
        'Particle Swarm Optimization': PSO,
        'Particle Swarm Optimization (PSO)': PSO,
        'GeneticAlgorithm': GeneticAlgorithm,
        'Genetic Algorithm': GeneticAlgorithm,
        'Genetic Algorithm (GA)': GeneticAlgorithm,
        'GA': GeneticAlgorithm,
    }
    
    method_class = method_map.get(recommendation.selected_method)
    if not method_class:
        raise ValueError(f"Unknown method: {recommendation.selected_method}")
    
    return method_class(**recommendation.parameters)


def prepare_problem_data(problem):
    return {
        'objective_function': problem.evaluate,
        'bounds': [(problem.lower_bounds[i], problem.upper_bounds[i]) 
                   for i in range(problem.dimension)],
        'dimension': problem.dimension,
    }


def run_single_experiment(method, problem, run_number, logger):
    method_name = method.__class__.__name__
    print(f"  Run {run_number + 1}: {method_name}...", end=" ", flush=True)
    
    problem.reset_evaluations()
    start_time = time.time()
    
    try:
        problem_data = prepare_problem_data(problem)
        
        result = method.fit(problem_data, callback=standard_progress_callback)
        
        computation_time = time.time() - start_time
        
        best_solution = result.get('best_solution')
        best_fitness = result.get('best_fitness')
        convergence_history = result.get('convergence_history', [])
        
        if best_fitness is None:
            print(f"ERROR: No fitness value returned")
            return None
        
        gap_percent = None
        if problem.optimal_value is not None:
            gap_percent = compute_gap_percentage(best_fitness, problem.optimal_value)
        
        print(f"Fitness: {best_fitness:.6f}", end="")
        if gap_percent is not None:
            print(f" | Error: {abs(best_fitness - problem.optimal_value):.6f} | Gap: {gap_percent:.4f}%", end="")
        print(f" | Time: {computation_time:.2f}s | Evals: {problem.function_evaluations}")
        
        return {
            'run': run_number + 1,
            'best_fitness': best_fitness,
            'best_solution': best_solution.tolist() if isinstance(best_solution, np.ndarray) else best_solution,
            'computation_time': computation_time,
            'convergence_history': convergence_history,
            'gap_percent': gap_percent,
            'iterations': len(convergence_history),
            'function_evaluations': problem.function_evaluations,
            'error': abs(best_fitness - problem.optimal_value) if problem.optimal_value is not None else None
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_experiments_with_config(method_class, method_params, problem, n_runs, logger, iteration_label="Initial"):
    print(f"\n{iteration_label} - Running {n_runs} independent experiments...")
    print("-" * 80)
    
    results = []
    for run in range(n_runs):
        method = method_class(**method_params)
        result = run_single_experiment(method, problem, run, logger)
        
        if result is not None:
            results.append(result)
    
    return results


def run_agent_optimization(agent, problem, memory_manager, n_runs=10, enable_feedback=True, max_feedback_iterations=2, logger=None):
    print(f"\n{'#'*80}")
    print(f"# Agent-Guided Optimization: {problem.problem_name}")
    print(f"# Function: {problem.function_name}, Dimension: {problem.dimension}")
    print(f"# Optimal Value: {problem.optimal_value}")
    print(f"# Feedback Loop: {'ENABLED' if enable_feedback else 'DISABLED'}")
    print(f"{'#'*80}")
    
    available_methods = {
        'PSO': PSO.PARAM_SPECS,
        'GeneticAlgorithm': GeneticAlgorithm.PARAM_SPECS,
    }

    recommendation = get_llm_recommendation(agent, problem, memory_manager)
    if recommendation is None:
        print("ERROR: Failed to get recommendation. Skipping this problem.")
        return None
    
    try:
        method_template = create_method_from_recommendation(recommendation)
        method_class = method_template.__class__
        method_params = recommendation.parameters
    except Exception as e:
        print(f"ERROR: Failed to create method: {e}")
        return None
    
    all_iterations = []
    
    print(f"\n{'='*80}")
    print(f"ITERATION 0: Initial Recommendation")
    print(f"{'='*80}")
    
    results = run_experiments_with_config(
        method_class, method_params, problem, n_runs, logger, 
        iteration_label="Initial Recommendation"
    )
    
    if not results:
        print("ERROR: No successful runs. Skipping this problem.")
        return None
    
    iteration_summary = compute_iteration_summary(
        results, problem, method_class, recommendation, iteration_num=0
    )
    all_iterations.append(iteration_summary)
    print_iteration_summary(iteration_summary, n_runs)
    
    # Plot convergence for initial iteration
    try:
        figures_dir = project_root / "outputs" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = problem.problem_name.replace(' ', '_').replace('-', '_')
        
        # Plot best run convergence
        plot_convergence(
            iteration_summary['best_run']['convergence_history'],
            title=f"{problem.problem_name} - Iteration 0 - Best Run Convergence",
            xlabel="Iteration",
            ylabel="Fitness Value",
            save_path=str(figures_dir / f"{safe_name}_iter0_convergence_{timestamp}.png"),
            show=False
        )
        print(f"  ✓ Saved convergence plot: {safe_name}_iter0_convergence_{timestamp}.png")
        
        # Plot all runs for this iteration
        iter_conv_data = {f"Run {r['run']}": r['convergence_history'] 
                         for r in iteration_summary['all_runs']}
        plot_multiple_convergence(
            iter_conv_data,
            title=f"{problem.problem_name} - Iteration 0 - All Runs",
            xlabel="Iteration",
            ylabel="Fitness Value",
            save_path=str(figures_dir / f"{safe_name}_iter0_all_runs_{timestamp}.png"),
            show=False
        )
        print(f"  ✓ Saved all-runs plot: {safe_name}_iter0_all_runs_{timestamp}.png")
    except Exception as e:
        print(f"  Warning: Could not generate iteration 0 convergence plots: {e}")

    best_fitness = iteration_summary['best_fitness']['min']

    memory_entry = {
        "problem": problem.problem_name,
        "Method": recommendation.selected_method,
        "Parameters": recommendation.parameters,
        "Fitness": best_fitness, 
        "Timestamp": datetime.now().isoformat()
    }
    print(f"   [Memory] Saving initial result (Fitness: {best_fitness:.6f})")
    memory_manager.save_memory("continuous_optimization", memory_entry)
    
    # Step 6: Get LLM interpretation of initial results
    try:
        print(f"\n{'='*80}")
        print(f"STEP 6: LLM Result Interpretation")
        print(f"{'='*80}")
        initial_interpretation = agent.interpret_results(
            problem_info=problem.get_info(),
            execution_result={
                'best_fitness': iteration_summary['best_fitness']['min'],
                'execution_time': iteration_summary['computation_time']['mean'],
                'iterations': int(iteration_summary['function_evaluations']['mean']),
                'metrics': {
                    'gap_percentage': iteration_summary['gap_percent']['mean'] if iteration_summary['gap_percent'] else None,
                    'error': iteration_summary['error']['mean'] if iteration_summary['error'] else None,
                }
            },
            recommendation={
                'selected_method': recommendation.selected_method,
                'parameters': recommendation.parameters,
                'expected_performance': recommendation.expected_performance,
            }
        )
        
        print(f"\n Performance Assessment: {initial_interpretation['performance_assessment'].upper()}")
        print(f"Confidence: {initial_interpretation['confidence_assessment']}")
        print(f"\nAnalysis:")
        print(f"{initial_interpretation['performance_explanation']}")
        print(f"\n Comparison with Expected:")
        print(f"{initial_interpretation['comparison_with_expected']}")
        
        if initial_interpretation['improvement_recommendations']:
            print(f"\n Improvement Recommendations:")
            for i, rec in enumerate(initial_interpretation['improvement_recommendations'], 1):
                print(f"  {i}. [{rec['type'].upper()}] {rec['suggestion']}")
        
        if initial_interpretation['next_steps']:
            print(f"\n🎯 Next Steps:")
            for i, step in enumerate(initial_interpretation['next_steps'], 1):
                print(f"  {i}. {step}")
        
        print(f"{'='*80}")
        iteration_summary['interpretation'] = initial_interpretation
    except Exception as e:
        print(f"Could not generate initial result interpretation: {e}")
    
    if enable_feedback and max_feedback_iterations > 0:
        current_recommendation = recommendation
        current_results = results
        
        for feedback_iter in range(1, max_feedback_iterations + 1):
            print(f"\n{'='*80}")
            print(f"ITERATION {feedback_iter}: Feedback Loop")
            print(f"{'='*80}")
            
            previous_result = {
                'best_fitness': iteration_summary['best_fitness'],
                'mean_fitness': iteration_summary['best_fitness']['mean'],
                'convergence_history': iteration_summary['best_run']['convergence_history'],
                'computation_time': iteration_summary['computation_time']['mean'],
                'parameters_used': current_recommendation.parameters,
            }
            
            previous_recommendation_dict = {
                'selected_method': current_recommendation.selected_method,
                'parameters': current_recommendation.parameters,
                'confidence': current_recommendation.confidence,
                'reasoning': current_recommendation.reasoning,
            }
            
            print(f"\nRequesting feedback from agent...")
            try:
                feedback_recommendation = agent.get_feedback_recommendation(
                    problem_info=problem.get_info(),
                    available_methods=available_methods,
                    previous_result=previous_result,
                    previous_recommendation=previous_recommendation_dict
                )
                
                print(f"\nFeedback Recommendation:")
                print(f"  Method: {feedback_recommendation.selected_method}")
                print(f"  Confidence: {feedback_recommendation.confidence:.2%}")
                print(f"  Reasoning: {feedback_recommendation.reasoning}")
                print(f"\n  Adjusted Parameters:")
                for param, value in feedback_recommendation.parameters.items():
                    old_value = current_recommendation.parameters.get(param, 'N/A')
                    changed = "🔸" if value != old_value else "  "
                    print(f"    {changed} {param}: {old_value} → {value}")
                
            except Exception as e:
                print(f"Error getting feedback: {e}")
                break
            
            try:
                feedback_method_template = create_method_from_recommendation(feedback_recommendation)
                feedback_method_class = feedback_method_template.__class__
                feedback_method_params = feedback_recommendation.parameters
            except Exception as e:
                print(f"Failed to create method from feedback: {e}")
                break
            
            # Run experiments with feedback parameters
            feedback_results = run_experiments_with_config(
                feedback_method_class, feedback_method_params, problem, n_runs, logger,
                iteration_label=f"Feedback Iteration {feedback_iter}"
            )
            
            if not feedback_results:
                print("Feedback iteration failed. Stopping feedback loop.")
                break
            
            feedback_summary = compute_iteration_summary(
                feedback_results, problem, feedback_method_class, 
                feedback_recommendation, iteration_num=feedback_iter
            )
            all_iterations.append(feedback_summary)
            print_iteration_summary(feedback_summary, n_runs)
            
            # Plot convergence for feedback iteration
            try:
                figures_dir = project_root / "outputs" / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = problem.problem_name.replace(' ', '_').replace('-', '_')
                
                # Plot best run convergence
                plot_convergence(
                    feedback_summary['best_run']['convergence_history'],
                    title=f"{problem.problem_name} - Iteration {feedback_iter} - Best Run Convergence",
                    xlabel="Iteration",
                    ylabel="Fitness Value",
                    save_path=str(figures_dir / f"{safe_name}_iter{feedback_iter}_convergence_{timestamp}.png"),
                    show=False
                )
                print(f"  ✓ Saved convergence plot: {safe_name}_iter{feedback_iter}_convergence_{timestamp}.png")
                
                # Plot all runs for this iteration
                iter_conv_data = {f"Run {r['run']}": r['convergence_history'] 
                                 for r in feedback_summary['all_runs']}
                plot_multiple_convergence(
                    iter_conv_data,
                    title=f"{problem.problem_name} - Iteration {feedback_iter} - All Runs",
                    xlabel="Iteration",
                    ylabel="Fitness Value",
                    save_path=str(figures_dir / f"{safe_name}_iter{feedback_iter}_all_runs_{timestamp}.png"),
                    show=False
                )
                print(f"  ✓ Saved all-runs plot: {safe_name}_iter{feedback_iter}_all_runs_{timestamp}.png")
            except Exception as e:
                print(f"  Warning: Could not generate iteration {feedback_iter} convergence plots: {e}")

            fb_best_fitness = feedback_summary['best_fitness']['min']
            fb_entry = {
                "problem": problem.problem_name,
                "Method": feedback_recommendation.selected_method,
                "Parameters": feedback_recommendation.parameters,
                "Fitness": fb_best_fitness, 
                "Timestamp": datetime.now().isoformat()
            }
            print(f"   [Memory] Saving feedback result (Fitness: {fb_best_fitness:.6f})")
            memory_manager.save_memory("continuous_optimization", fb_entry)
            
            improvement = iteration_summary['best_fitness']['mean'] - feedback_summary['best_fitness']['mean']
            improvement_pct = (improvement / iteration_summary['best_fitness']['mean']) * 100 if iteration_summary['best_fitness']['mean'] != 0 else 0
            
            print(f"\nImprovement Analysis:")
            print(f"  Previous Mean: {iteration_summary['best_fitness']['mean']:.6f}")
            print(f"  Current Mean:  {feedback_summary['best_fitness']['mean']:.6f}")
            print(f"  Absolute Improvement: {improvement:.6f}")
            print(f"  Percentage Improvement: {improvement_pct:.2f}%")
            
            if improvement > 0:
                print(f"  Performance IMPROVED!")
            else:
                print(f"  Performance did not improve.")
            
            current_recommendation = feedback_recommendation
            current_results = feedback_results
            iteration_summary = feedback_summary
    
    best_iteration_idx = np.argmin([iter_sum['best_fitness']['mean'] for iter_sum in all_iterations])
    best_iteration = all_iterations[best_iteration_idx]
    
    if len(all_iterations) > 1:
        try:
            print(f"\n{'='*80}")
            print(f"FINAL STEP 6: LLM Interpretation of Best Results")
            print(f"{'='*80}")
            final_interpretation = agent.interpret_results(
                problem_info=problem.get_info(),
                execution_result={
                    'best_fitness': best_iteration['best_fitness']['min'],
                    'execution_time': best_iteration['computation_time']['mean'],
                    'iterations': int(best_iteration['function_evaluations']['mean']),
                    'metrics': {
                        'gap_percentage': best_iteration['gap_percent']['mean'] if best_iteration['gap_percent'] else None,
                        'error': best_iteration['error']['mean'] if best_iteration['error'] else None,
                    }
                },
                recommendation={
                    'selected_method': best_iteration['recommendation']['selected_method'],
                    'parameters': best_iteration['recommendation']['parameters'],
                    'expected_performance': best_iteration['recommendation']['expected_performance'],
                }
            )
            
            print(f"\n Final Assessment: {final_interpretation['performance_assessment'].upper()}")
            print(f"Confidence: {final_interpretation['confidence_assessment']}")
            print(f"\n After {len(all_iterations)} iterations of optimization:")
            print(f"{final_interpretation['performance_explanation']}")
            print(f"{'='*80}")
            best_iteration['interpretation'] = final_interpretation
        except Exception as e:
            print(f"Could not generate final result interpretation: {e}")
    
    summary = {
        'problem_name': problem.problem_name,
        'function_name': problem.function_name,
        'dimension': problem.dimension,
        'optimal_value': problem.optimal_value,
        'feedback_enabled': enable_feedback,
        'total_iterations': len(all_iterations),
        'best_iteration': best_iteration_idx,
        'all_iterations': all_iterations,
        'final_best_fitness': best_iteration['best_fitness'],
        'final_best_run': best_iteration['best_run'],
        'final_interpretation': best_iteration.get('interpretation'),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY: {problem.problem_name}")
    print(f"{'='*80}")
    print(f"Total Iterations: {len(all_iterations)}")
    print(f"Best Iteration: {best_iteration_idx}")
    print(f"Best Mean Fitness: {best_iteration['best_fitness']['mean']:.6f}")
    print(f"Overall Best Fitness: {best_iteration['best_fitness']['min']:.6f}")
    
    if len(all_iterations) > 1:
        initial_mean = all_iterations[0]['best_fitness']['mean']
        final_mean = best_iteration['best_fitness']['mean']
        total_improvement = initial_mean - final_mean
        total_improvement_pct = (total_improvement / initial_mean) * 100 if initial_mean != 0 else 0
        print(f"\nOverall Improvement from Initial:")
        print(f"  Initial Mean: {initial_mean:.6f}")
        print(f"  Final Mean:   {final_mean:.6f}")
        print(f"  Total Improvement: {total_improvement:.6f} ({total_improvement_pct:.2f}%)")
    
    print(f"{'='*80}\n")
    
    return summary


def compute_iteration_summary(results, problem, method_class, recommendation, iteration_num):
    fitness_values = [r['best_fitness'] for r in results]
    times = [r['computation_time'] for r in results]
    errors = [r['error'] for r in results if r['error'] is not None]
    gaps = [r['gap_percent'] for r in results if r['gap_percent'] is not None]
    evaluations = [r['function_evaluations'] for r in results]
    
    fitness_stats = compute_statistics(fitness_values)
    time_stats = compute_statistics(times)
    error_stats = compute_statistics(errors) if errors else None
    gap_stats = compute_statistics(gaps) if gaps else None
    eval_stats = compute_statistics(evaluations)
    
    best_run_idx = np.argmin(fitness_values)
    best_run = results[best_run_idx]
    
    return {
        'iteration': iteration_num,
        'method': method_class.__name__,
        'recommendation': {
            'selected_method': recommendation.selected_method,
            'confidence': recommendation.confidence,
            'expected_performance': recommendation.expected_performance,
            'reasoning': recommendation.reasoning,
            'parameters': recommendation.parameters,
        },
        'n_runs': len(results),
        'best_fitness': fitness_stats,
        'error': error_stats,
        'gap_percent': gap_stats,
        'computation_time': time_stats,
        'function_evaluations': eval_stats,
        'best_run': best_run,
        'all_runs': results,
    }


def print_iteration_summary(summary, n_runs):
    print(f"\n{'='*80}")
    print(f"Iteration {summary['iteration']} Summary")
    print(f"{'='*80}")
    print(f"Method: {summary['method']}")
    print(f"Successful Runs: {summary['n_runs']}/{n_runs}")
    print(f"\nBest Fitness:")
    print(f"  Best:   {summary['best_fitness']['min']:.6f}")
    print(f"  Mean:   {summary['best_fitness']['mean']:.6f} ± {summary['best_fitness']['std']:.6f}")
    print(f"  Median: {summary['best_fitness']['median']:.6f}")
    
    if summary['error']:
        print(f"\nError from Optimal:")
        print(f"  Best: {summary['error']['min']:.6f}")
        print(f"  Mean: {summary['error']['mean']:.6f} ± {summary['error']['std']:.6f}")
    
    if summary['gap_percent']:
        print(f"\nGap Percentage:")
        print(f"  Best: {summary['gap_percent']['min']:.4f}%")
        print(f"  Mean: {summary['gap_percent']['mean']:.4f}%")
    
    print(f"\nComputation Time: {summary['computation_time']['mean']:.2f}s ± {summary['computation_time']['std']:.2f}s")
    print(f"Function Evaluations: {summary['function_evaluations']['mean']:.0f}")
    print(f"{'='*80}")

def perform_statistical_analysis(all_results, output_dir):
    """Perform Wilcoxon statistical comparisons between optimization methods."""
    if len(all_results) < 2:
        return
    
    print("\n" + "="*100)
    print("STATISTICAL ANALYSIS: WILCOXON PAIRWISE COMPARISONS")
    print("="*100)
    
    # Group results by problem and iteration
    problems_methods = {}
    
    for result in all_results:
        problem_name = result['problem_name']
        best_iter = result['all_iterations'][result['best_iteration']]
        
        if problem_name not in problems_methods:
            problems_methods[problem_name] = {}
        
        method_name = best_iter['method']
        fitness_values = [r['best_fitness'] for r in best_iter['all_runs']]
        problems_methods[problem_name][method_name] = fitness_values
    
    # Perform comparisons for each problem
    statistical_results = {}
    
    for problem, methods_data in problems_methods.items():
        if len(methods_data) < 2:
            continue
        
        print(f"\nProblem: {problem}")
        comparison_results = pairwise_wilcoxon_comparison(methods_data)
        statistical_results[problem] = comparison_results
        
        print_wilcoxon_summary(comparison_results, title=f"Wilcoxon Test Results for {problem}")
    
    # Save statistical results to CSV
    csv_path = output_dir / f"statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, 'w') as f:
        f.write("Problem,Method1,Method2,P_Value,Significant,Effect_Size\n")
        
        for problem, comp_results in statistical_results.items():
            methods = comp_results['methods']
            p_values = comp_results['p_values']
            significant = comp_results['significant']
            effect_sizes = comp_results['effect_sizes']
            
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    p_val = p_values[i, j]
                    sig = "Yes" if significant[i, j] else "No"
                    eff = effect_sizes[i, j]
                    
                    if not np.isnan(p_val):
                        f.write(f"{problem},{methods[i]},{methods[j]},{p_val:.6f},{sig},{eff:.4f}\n")
    
    print(f"\nStatistical results saved to: {csv_path}")


def main():
    logger = get_experiment_logger("function_optimization", str(project_root / "outputs" / "logs"))
    logger.info("="*80)
    logger.info("MetaMind Function Optimization Benchmark")
    logger.info("="*80)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        return
    
    print("Initializing MetaMind Agent...")
    agent = MetaMindAgent(
        api_key=api_key,
        model="NousResearch/Hermes-3-Llama-3.1-70B",
        temperature=0.3,
        verbose=True
    )
    print("Agent initialized successfully!\n")
    
    benchmark_problems = [
        ('rastrigin', 10),
        ('ackley', 10),
        ('rosenbrock', 10),
        ('sphere', 10),
        
        ('rastrigin', 20),
        ('ackley', 20),
        ('rosenbrock', 20),
        ('sphere', 20),

        
        ('rastrigin', 30),
        ('ackley', 30),
        ('rosenbrock', 30),
        ('sphere', 30),


    ]
    
    all_results = []
    n_runs = 5
    enable_feedback = True
    max_feedback_iterations = 2
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Runs per iteration: {n_runs}")
    print(f"Feedback loop: {'ENABLED ✓' if enable_feedback else 'DISABLED'}")
    print(f"Max feedback iterations: {max_feedback_iterations}")
    print(f"{'='*80}\n")

    memory_dir = project_root / "outputs" / "memory"
    memory_manager = MemoryManager(output_dir=memory_dir)
    print(f"Memory Manager initialized. Saving to: {memory_manager.output_dir.absolute()}")
    
    for func_name, dimension in benchmark_problems:
        try:
            problem = create_benchmark_function(func_name, dimension)
            
            result = run_agent_optimization(
                agent, problem, 
                memory_manager=memory_manager, 
                n_runs=n_runs, 
                enable_feedback=enable_feedback,
                max_feedback_iterations=max_feedback_iterations,
                logger=logger
            )
            
            if result is not None:
                all_results.append(result)
            
        except Exception as e:
            print(f"Error with {func_name}-{dimension}D: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_results:
        output_dir = project_root / "outputs" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"function_optimization_agent_{timestamp}.json"
        
        serializable_results = convert_to_serializable(all_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Perform statistical analysis
        perform_statistical_analysis(all_results, output_dir)
        
        print("\nGenerating comparison plots...")
        try:
            figures_dir = project_root / "outputs" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            convergence_data = {}
            for result in all_results:
                label = f"{result['function_name']}-{result['dimension']}D"
                convergence_data[label] = result['final_best_run']['convergence_history']
            
            if convergence_data:
                plot_multiple_convergence(
                    convergence_data,
                    title="Agent-Guided Function Optimization (with Feedback) - Best Runs",
                    ylabel="Fitness Value",
                    save_path=str(figures_dir / f"function_opt_convergence_{timestamp}.png"),
                    show=False
                )
            
            box_data = {}
            for result in all_results:
                label = f"{result['function_name']}-{result['dimension']}D"
                best_iter = result['all_iterations'][result['best_iteration']]
                fitness_values = [r['best_fitness'] for r in best_iter['all_runs']]
                box_data[label] = fitness_values
            
            if box_data:
                plot_box_comparison(
                    box_data,
                    title="Agent-Guided Optimization (with Feedback) - Fitness Distribution",
                    ylabel="Best Fitness",
                    save_path=str(figures_dir / f"function_opt_boxplot_{timestamp}.png"),
                    show=False
                )
            
            # Plot convergence with confidence bands for all runs (grouped by problem)
            try:
                # Group by problem name (without dimension)
                problems_data = {}
                for result in all_results:
                    problem_base_name = result['function_name']  # e.g., "Rastrigin Function"
                    
                    if problem_base_name not in problems_data:
                        problems_data[problem_base_name] = {}
                    
                    label = f"{result['dimension']}D"
                    best_iter = result['all_iterations'][result['best_iteration']]
                    all_convergence_histories = [r['convergence_history'] for r in best_iter['all_runs']]
                    problems_data[problem_base_name][label] = all_convergence_histories
                
                # Create a plot for each problem showing all dimensions
                for problem_name, dimensions_data in problems_data.items():
                    if dimensions_data:
                        safe_problem_name = problem_name.replace(' ', '_').lower()
                        plot_convergence_with_bands(
                            dimensions_data,
                            title=f"{problem_name} - Convergence with 95% CI (All Dimensions)",
                            xlabel="Iteration",
                            ylabel="Fitness Value",
                            confidence=0.95,
                            save_path=str(figures_dir / f"{safe_problem_name}_convergence_bands_{timestamp}.png"),
                            show=False
                        )
            except Exception as e:
                print(f"Warning: Could not generate convergence bands plots: {e}")
            
            # Plot comparison of iterations for each problem
            if enable_feedback:
                for result in all_results:
                    if result['total_iterations'] > 1:
                        # Feedback loop progress (mean fitness across iterations)
                        iter_means = [iter_data['best_fitness']['mean'] 
                                     for iter_data in result['all_iterations']]
                        
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(8, 6))
                        plt.plot(range(len(iter_means)), iter_means, 
                                marker='o', linewidth=2, markersize=8, color='#2E86AB')
                        plt.xlabel('Iteration', fontsize=12)
                        plt.ylabel('Mean Best Fitness', fontsize=12)
                        plt.title(f"Feedback Loop Progress: {result['problem_name']}", 
                                fontsize=14, fontweight='bold')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        safe_name = result['problem_name'].replace(' ', '_').replace('-', '_')
                        plt.savefig(str(figures_dir / f"feedback_progress_{safe_name}_{timestamp}.png"), 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        # Compare best run convergence across all iterations
                        iteration_convergence = {}
                        for iter_data in result['all_iterations']:
                            iter_num = iter_data['iteration']
                            iteration_convergence[f"Iteration {iter_num}"] = iter_data['best_run']['convergence_history']
                        
                        plot_multiple_convergence(
                            iteration_convergence,
                            title=f"{result['problem_name']} - Best Run per Iteration",
                            xlabel="Algorithm Iteration",
                            ylabel="Fitness Value",
                            save_path=str(figures_dir / f"{safe_name}_iterations_comparison_{timestamp}.png"),
                            show=False
                        )
            
            print(f"Plots saved to: {figures_dir}")
            
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"Total Problems Tested: {len(all_results)}")
    print(f"Feedback Loop: {'ENABLED ✓' if enable_feedback else 'DISABLED'}")
    
    if all_results:
        total_initial_runs = sum(len(r['all_iterations'][0]['all_runs']) for r in all_results)
        total_all_runs = sum(sum(len(it['all_runs']) for it in r['all_iterations']) for r in all_results)
        print(f"Total Runs: {total_all_runs} (across all iterations)")
        
        print("\nBest Results per Problem:")
        print("-" * 80)
        for result in all_results:
            best_iter = result['all_iterations'][result['best_iteration']]
            iter_label = f"[Iter {result['best_iteration']}]" if result['total_iterations'] > 1 else ""
            
            improvement = ""
            if result['total_iterations'] > 1:
                initial_mean = result['all_iterations'][0]['best_fitness']['mean']
                final_mean = result['final_best_fitness']['mean']
                impr_pct = ((initial_mean - final_mean) / initial_mean * 100) if initial_mean != 0 else 0
                improvement = f" | Improvement: {impr_pct:+.1f}%"
            
            print(f"{result['problem_name']:20s} {iter_label:10s} | "
                  f"Method: {best_iter['method']:15s} | "
                  f"Best: {result['final_best_fitness']['min']:.6f} | "
                  f"Mean: {result['final_best_fitness']['mean']:.6f}{improvement}")
    
    print("="*80)
    
    agent_stats = agent.get_stats()
    print(f"\nAgent Statistics:")
    print(f"  Total LLM Calls: {agent_stats['total_calls']}")
    print(f"  Model: {agent_stats['model']}")
    print(f"  Temperature: {agent_stats['temperature']}")
    
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()