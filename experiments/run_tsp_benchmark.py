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

from src.problems.tsp import TSPProblem, load_tsplib_instance, create_random_tsp
from src.methods.evolutionary.aco import AntColonyOptimization
from src.methods.evolutionary.ga import GeneticAlgorithm
from src.methods.evolutionary.gp import GeneticProgramming
from src.methods.evolutionary.pso import PSO
from src.methods.neural.mlp import MLP
from src.methods.neural.perceptron import Perceptron
from src.methods.neural.hopfield import HopfieldNetwork
from src.methods.neural.som import SOM
from src.methods.fuzzy.controller import FuzzyController
from src.orchestrator.agent import MetaMindAgent
from src.utils.logging import setup_logger, get_experiment_logger, standard_progress_callback
from src.utils.metrics import compute_statistics , compute_gap_percentage
from src.utils.plotting import plot_convergence, plot_comparison_table , plot_box_comparison


def get_llm_recommendation(agent, problem):

    print(f"\nAsking LLM for recommendation on {problem.problem_name}...")

    problem_info = problem.get_info()
    
    if isinstance(problem, TSPProblem):
        available_methods = {
            'AntColonyOptimization': AntColonyOptimization.PARAM_SPECS,
            'GeneticAlgorithm': GeneticAlgorithm.PARAM_SPECS,
        }
    else:
        # For other problem types, offer all methods
        available_methods = {
            # Evolutionary/Swarm Intelligence
            'AntColonyOptimization': AntColonyOptimization.PARAM_SPECS,
            'GeneticAlgorithm': GeneticAlgorithm.PARAM_SPECS,
            'GeneticProgramming': GeneticProgramming.PARAM_SPECS,
            'PSO': PSO.PARAM_SPECS,
            # Neural Networks
            'MLP': MLP.PARAM_SPECS,
            'Perceptron': Perceptron.PARAM_SPECS,
            'HopfieldNetwork': HopfieldNetwork.PARAM_SPECS,
            'SOM': SOM.PARAM_SPECS,
            # Fuzzy Systems
            'FuzzyController': FuzzyController.PARAM_SPECS,
        }
    

    try:
        recommendation = agent.get_recommendation(
            problem_info=problem_info,
            available_methods=available_methods,
            context="This is a TSP benchmark experiment. We need the best-performing method."
        )
        
        print(f"\n LLM Recommendation:")
        print(f"  Method: {recommendation.selected_method}")
        print(f"  Confidence: {recommendation.confidence:.2%}")
        print(f"  Expected Performance: {recommendation.expected_performance}")
        print(f"  Reasoning: {recommendation.reasoning}")
        print(f"  Parameters: {json.dumps(recommendation.parameters, indent=4)}")
        
        if recommendation.alternative_methods:
            print(f"  Alternatives: {', '.join(recommendation.alternative_methods)}")
        
        if recommendation.warnings:
            print(f"Warnings:")
            for warning in recommendation.warnings:
                print(f"    - {warning}")
        
        return recommendation
        
    except Exception as e:
        print(f"Error getting LLM recommendation: {e}")
        return None


def create_method_from_recommendation(recommendation):
    method_map = {
        'AntColonyOptimization': AntColonyOptimization,
        'ACO': AntColonyOptimization,
        'GeneticAlgorithm': GeneticAlgorithm,
        'GA': GeneticAlgorithm,
        'GeneticProgramming': GeneticProgramming,
        'GP': GeneticProgramming,
        'PSO': PSO,
        'ParticleSwarmOptimization': PSO,
        'MLP': MLP,
        'MultiLayerPerceptron': MLP,
        'Perceptron': Perceptron,
        'HopfieldNetwork': HopfieldNetwork,
        'Hopfield': HopfieldNetwork,
        'SOM': SOM,
        'SelfOrganizingMap': SOM,
        'FuzzyController': FuzzyController,
        'Fuzzy': FuzzyController,
    }
    
    method_class = method_map.get(recommendation.selected_method)
    if not method_class:
        raise ValueError(f"Unknown method: {recommendation.selected_method}")
    
    return method_class(**recommendation.parameters)


def run_single_experiment(method, problem, run_number):
    method_name = method.__class__.__name__
    print(f"  Run {run_number + 1}: {method_name}...", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        result = method.fit(problem, callback=standard_progress_callback)
        
        computation_time = time.time() - start_time
        
        best_solution = result.get('best_solution') or result.get('best_tour')
        best_fitness = result.get('best_fitness') or result.get('best_length')
        convergence_history = result.get('convergence_history', [])
        
        if best_fitness is None:
            print(f"ERROR: No fitness value returned")
            return None
        
        gap_percent = None
        if problem.optimal_value is not None and best_fitness > 0:
            gap_percent = compute_gap_percentage(best_fitness, problem.optimal_value)
        
        print(f"Distance: {best_fitness:.2f}", end="")
        if gap_percent is not None:
            print(f" (Gap: {gap_percent:.2f}%)", end="")
        print(f" Time: {computation_time:.2f}s")
        
        return {
            'run': run_number + 1,
            'best_fitness': best_fitness,
            'best_solution': best_solution,
            'computation_time': computation_time,
            'convergence_history': convergence_history,
            'gap_percent': gap_percent,
            'iterations': len(convergence_history)
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None


def run_method_on_problem(method_class, method_params, problem, n_runs=5):
    print(f"\n{method_class.__name__} on {problem.problem_name}")
    print("=" * 70)
    
    results = []
    
    for run in range(n_runs):
        method = method_class(**method_params)
        result = run_single_experiment(method, problem, run)
        
        if result is not None:
            results.append(result)
    
    if not results:
        return None
    
    fitness_values = [r['best_fitness'] for r in results]
    times = [r['computation_time'] for r in results]
    gaps = [r['gap_percent'] for r in results if r['gap_percent'] is not None]
    
    # Use compute_statistics utility
    fitness_stats = compute_statistics(fitness_values)
    time_stats = compute_statistics(times)
    gap_stats = compute_statistics(gaps) if gaps else None
    
    stats = {
        'method': method_class.__name__,
        'problem': problem.problem_name,
        'n_runs': len(results),
        'best_fitness': fitness_stats,
        'computation_time': time_stats,
        'gap_percent': gap_stats,
        'optimal_value': problem.optimal_value,
        'all_runs': results
    }
    
    # Compute confidence intervals
    from src.utils.metrics import compute_confidence_interval
    ci_lower, ci_upper = compute_confidence_interval(fitness_values)
    
    print(f"\nSummary Statistics:")
    print(f"  Best Tour Length: {stats['best_fitness']['min']:.2f}")
    print(f"  Mean Tour Length: {stats['best_fitness']['mean']:.2f} ± {stats['best_fitness']['std']:.2f}")
    if ci_lower and ci_upper:
        print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  Mean Time: {stats['computation_time']['mean']:.2f}s ± {stats['computation_time']['std']:.2f}s")
    if stats['gap_percent']:
        print(f"  Mean Gap to Optimal: {stats['gap_percent']['mean']:.2f}% ± {stats['gap_percent']['std']:.2f}%")
    
    # Plot convergence for best run
    best_run = min(results, key=lambda r: r['best_fitness'])
    if best_run['convergence_history']:
        from pathlib import Path
        figures_dir = Path(__file__).parent.parent / "outputs" / "figures"
        plot_path = figures_dir / f"convergence_{method_class.__name__}_{problem.problem_name}.png"
        plot_convergence(
            best_run['convergence_history'],
            title=f"{method_class.__name__} on {problem.problem_name}",
            ylabel="Tour Length",
            save_path=str(plot_path),
            show=False
        )
    
    return stats


def create_summary_table(all_results):
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Problem':<20} {'Method':<15} {'Best':<12} {'Mean±Std':<20} {'Gap%':<12} {'Time(s)':<12}")
    print("-" * 100)
    
    for result in all_results:
        if result is None:
            continue
        
        problem_name = result['problem']
        method_name = result['method']
        best = result['best_fitness']['min']
        mean = result['best_fitness']['mean']
        std = result['best_fitness']['std']
        time_mean = result['computation_time']['mean']
        
        gap_str = "N/A"
        if result['gap_percent']:
            gap_mean = result['gap_percent']['mean']
            gap_str = f"{gap_mean:.2f}%"
        
        mean_std_str = f"{mean:.2f}±{std:.2f}"
        print(f"{problem_name:<20} {method_name:<15} {best:<12.2f} {mean_std_str:<20} {gap_str:<12} {time_mean:<12.2f}")
    
    print("=" * 100)


def save_results(all_results, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_file = output_path / f"tsp_benchmark_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {json_file}")
    
    csv_file = output_path / f"tsp_benchmark_summary_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("Problem,Method,N_Runs,Best,Mean,Std,Median,Gap_Mean%,Gap_Std%,Time_Mean,Time_Std,Optimal\n")
        for result in all_results:
            if result is None:
                continue
            
            problem = result['problem']
            method = result['method']
            n_runs = result['n_runs']
            best = result['best_fitness']['min']
            mean = result['best_fitness']['mean']
            std = result['best_fitness']['std']
            median = result['best_fitness']['median']
            time_mean = result['computation_time']['mean']
            time_std = result['computation_time']['std']
            optimal = result['optimal_value'] if result['optimal_value'] else 'N/A'
            
            if result['gap_percent']:
                gap_mean = f"{result['gap_percent']['mean']:.2f}"
                gap_std = f"{result['gap_percent']['std']:.2f}"
            else:
                gap_mean = 'N/A'
                gap_std = 'N/A'
            
            f.write(f"{problem},{method},{n_runs},{best:.2f},{mean:.2f},{std:.2f},{median:.2f},")
            f.write(f"{gap_mean},{gap_std},{time_mean:.2f},{time_std:.2f},{optimal}\n")
    
    print(f"Summary CSV saved to: {csv_file}")


def main():
    # Setup logger
    from src.utils.logging import get_experiment_logger
    logger = get_experiment_logger("tsp_benchmark", str(project_root / "outputs" / "logs"))
    
    logger.info("="*100)
    logger.info("TSP BENCHMARK EXPERIMENT")
    logger.info("="*100)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set!")
        logger.error("Please set it with: export GROQ_API_KEY='your-api-key'")
        return
    
    agent = MetaMindAgent(api_key=api_key, verbose=True)
    logger.info(f"MetaMind Agent initialized with model: {agent.model}")
    
    n_runs = 5  # Number of runs per method per problem
    output_dir = project_root / "outputs" / "results"
    figures_dir = project_root / "outputs" / "figures"
    use_llm_selection = True  # Toggle LLM-based selection
    
    problems = []
    
    tsplib_dir = project_root / "data" / "tsplib"
    tsplib_instances = ['eil51', 'berlin52', 'kroA100']
    
    logger.info("Loading TSPLIB instances...")
    for instance_name in tsplib_instances:
        try:
            problem = load_tsplib_instance(instance_name, str(tsplib_dir))
            problems.append(problem)
            logger.info(f"  ✓ Loaded {instance_name}: {problem.n_cities} cities, optimal={problem.optimal_value}")
        except Exception as e:
            logger.warning(f"  ✗ Failed to load {instance_name}: {e}")
    
    logger.info("\nGenerating random instances...")
    # Random 30-city instance with exact solver
    logger.info("  Generating random_30 and computing exact optimal solution...")
    problem_30 = create_random_tsp(n_cities=30, seed=42, bounds=(0, 1000))
    try:
        optimal_tour, optimal_distance = problem_30.solve_exact_branch_and_bound(time_limit=300)
        problem_30.optimal_value = optimal_distance
        problems.append(problem_30)
        logger.info(f"  ✓ Generated random_30: {problem_30.n_cities} cities, optimal={optimal_distance:.2f}")
    except Exception as e:
        logger.warning(f"  ⚠ Failed to solve random_30 exactly: {e}")
        logger.info(f"  Using LKH estimation instead...")
        _, lkh_distance = problem_30.get_lkh_estimation(num_starts=30, use_2opt=True)
        problem_30.optimal_value = lkh_distance
        problems.append(problem_30)
        logger.info(f"  ✓ Generated random_30: {problem_30.n_cities} cities, LKH estimate={lkh_distance:.2f}")
    
    # Random 50-city instance with LKH estimation
    logger.info("  Generating random_50 and computing LKH estimation...")
    problem_50 = create_random_tsp(n_cities=50, seed=123, bounds=(0, 1000))
    _, lkh_distance = problem_50.get_lkh_estimation(num_starts=50, use_2opt=True, time_limit=120)
    problem_50.optimal_value = lkh_distance
    problems.append(problem_50)
    logger.info(f"  ✓ Generated random_50: {problem_50.n_cities} cities, LKH estimate={lkh_distance:.2f}")
    
    all_results = []
    
    for problem in problems:
        print(f"\n{'=' * 100}")
        print(f"PROBLEM: {problem.problem_name} ({problem.n_cities} cities)")
        if problem.optimal_value:
            print(f"Optimal Value: {problem.optimal_value}")
        print(f"{'=' * 100}")
        
        if use_llm_selection:
            recommendation = get_llm_recommendation(agent, problem)
            
            if recommendation:
                # Run the recommended method
                try:
                    method_class = {
                        # Evolutionary/Swarm Intelligence
                        'AntColonyOptimization': AntColonyOptimization,
                        'ACO': AntColonyOptimization,
                        'GeneticAlgorithm': GeneticAlgorithm,
                        'GA': GeneticAlgorithm,
                        'GeneticProgramming': GeneticProgramming,
                        'GP': GeneticProgramming,
                        'PSO': PSO,
                        'ParticleSwarmOptimization': PSO,
                        # Neural Networks
                        'MLP': MLP,
                        'MultiLayerPerceptron': MLP,
                        'Perceptron': Perceptron,
                        'HopfieldNetwork': HopfieldNetwork,
                        'Hopfield': HopfieldNetwork,
                        'SOM': SOM,
                        'SelfOrganizingMap': SOM,
                        # Fuzzy Systems
                        'FuzzyController': FuzzyController,
                        'Fuzzy': FuzzyController,
                    }.get(recommendation.selected_method)
                    
                    if method_class:
                        stats = run_method_on_problem(
                            method_class,
                            recommendation.parameters,
                            problem,
                            n_runs=n_runs
                        )
                        if stats:
                            stats['llm_confidence'] = recommendation.confidence
                            stats['llm_reasoning'] = recommendation.reasoning
                            stats['llm_expected_performance'] = recommendation.expected_performance
                            all_results.append(stats)
                except Exception as e:
                    print(f"Error running recommended method: {e}")
        else:
            # Original fixed method configuration
            methods = [
                {
                    'class': AntColonyOptimization,
                    'params': {
                        'n_ants': 50,
                        'max_iterations': 500,
                        'alpha': 1.0,
                        'beta': 2.5,
                        'evaporation_rate': 0.5,
                        'q': 100,
                        'local_search': True
                    }
                },
                {
                    'class': GeneticAlgorithm,
                    'params': {
                        'population_size': 100,
                        'generations': 500,
                        'crossover_rate': 0.8,
                        'mutation_rate': 0.1,
                        'selection': 'tournament',
                        'tournament_size': 5,
                        'elitism': 2
                    }
                }
            ]
            
            for method_config in methods:
                stats = run_method_on_problem(
                    method_config['class'],
                    method_config['params'],
                    problem,
                    n_runs=n_runs
                )
                if stats:
                    all_results.append(stats)
    
    # Generate summary
    create_summary_table(all_results)
    
    # Save results
    save_results(all_results, output_dir)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    if all_results:
        # Plot comparison table
        table_path = figures_dir / f"comparison_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_comparison_table(all_results, save_path=str(table_path))
        
        # Group results by problem for box plots
        problems_dict = {}
        for result in all_results:
            problem_name = result['problem']
            method_name = result['method']
            key = f"{problem_name}_{method_name}"
            fitness_values = [r['best_fitness'] for r in result['all_runs']]
            problems_dict[key] = fitness_values
        
        if len(problems_dict) > 1:
            box_path = figures_dir / f"box_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_box_comparison(
                problems_dict,
                title="Performance Comparison Across All Experiments",
                ylabel="Tour Length",
                save_path=str(box_path),
                show=False
            )
    
    # Print LLM agent stats
    agent_stats = agent.get_stats()
    logger.info("="*100)
    logger.info("LLM AGENT STATISTICS")
    logger.info("="*100)
    logger.info(f"Total LLM Calls: {agent_stats['total_calls']}")
    logger.info(f"Model: {agent_stats['model']}")
    logger.info(f"Temperature: {agent_stats['temperature']}")
    
    logger.info("="*100)
    logger.info("EXPERIMENT COMPLETED")
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total Problems: {len(problems)}")
    logger.info(f"Total Experiments: {len(all_results)}")
    logger.info("="*100)


if __name__ == "__main__":
    main()
