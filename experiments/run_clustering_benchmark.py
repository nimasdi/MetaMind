import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Import Framework Components
from src.problems.clustering import IrisProblem, MallCustomersProblem, SyntheticClusteringProblem
from src.orchestrator.agent import MetaMindAgent

# Import Utilities
from src.utils.logging import get_experiment_logger, standard_progress_callback
from src.utils.plotting import plot_box_comparison, plot_convergence, plot_convergence_with_bands
from src.utils.metrics import pairwise_wilcoxon_comparison, print_wilcoxon_summary

# Import Methods
from src.methods.neural.som import SOM
from src.methods.neural.mlp import MLP
from src.methods.neural.perceptron import Perceptron
from src.methods.neural.hopfield import HopfieldNetwork
from src.methods.evolutionary.ga import GeneticAlgorithm
from src.methods.evolutionary.gp import GeneticProgramming
from src.methods.evolutionary.pso import PSO
from src.methods.evolutionary.aco import AntColonyOptimization
from src.methods.fuzzy.controller import FuzzyController
from src.orchestrator.memory import MemoryManager

def get_method_class(method_name):
    """Maps LLM string selection to actual Python class."""
    mapping = {
        'SOM': SOM, 'SelfOrganizingMap': SOM, 'Kohonen': SOM,
        'GA': GeneticAlgorithm, 'GeneticAlgorithm': GeneticAlgorithm,
        'PSO': PSO, 'ParticleSwarmOptimization': PSO,
        'Fuzzy': FuzzyController, 'FuzzyController': FuzzyController,
    }
    return mapping.get(method_name, SOM)

def evaluate_clustering(X, labels, true_labels=None):
    """Computes clustering metrics safely."""
    # [CRITICAL FIX]: SOM predict returns (bmu_indices, distances).
    if isinstance(labels, tuple):
        labels = labels[0]
        
    labels = np.array(labels).astype(int).ravel()
    
    n_labels = len(np.unique(labels))
    n_samples = len(X)

    if n_labels < 2 or n_labels >= n_samples:
        return {
            'silhouette': -1.0, 
            'davies_bouldin': float('inf'), 
            'calinski_harabasz': 0.0,
            'ari': 0.0, 
            'nmi': 0.0, 
            'n_clusters': n_labels
        }

    try:
        metrics = {
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'n_clusters': n_labels
        }
    except Exception as e:
        print(f"Metric calculation error: {e}")
        return {'silhouette': -1.0, 'davies_bouldin': float('inf'), 'n_clusters': n_labels}
    
    if true_labels is not None:
        metrics['ari'] = adjusted_rand_score(true_labels, labels)
        metrics['nmi'] = normalized_mutual_info_score(true_labels, labels)
    else:
        metrics['ari'] = None
        metrics['nmi'] = None
        
    return metrics

def print_llm_json_style(rec):
    output = {
        "problem_type": "clustering",
        "selected_method": rec.selected_method,
        "reasoning": rec.reasoning[:120] + "...", 
        "parameters": rec.parameters,
        "backup_method": rec.alternative_methods[0] if rec.alternative_methods else "None",
        "confidence": rec.confidence
    }
    print("-" * 60)
    print("LLM output format:")
    print(json.dumps(output, indent=4))
    print("-" * 60)

def print_feedback_analysis(interpretation, metrics, previous_best):
    print("\nLLM feedback output:")
    print("## Results Analysis")
    score = metrics.get('silhouette', -1)
    imp_str = ""
    if previous_best != -1.0:
        diff = score - previous_best
        imp_str = f"(Change: {diff:+.4f})"

    print(f"The method achieved a Silhouette Score of {score:.4f} {imp_str}.")
    print(f"Assessment: {interpretation.get('performance_assessment', 'N/A').upper()}")
    print("### Observations:")
    print(f"- {interpretation.get('performance_explanation', 'No explanation provided.')}")
    print("### Recommendations:")
    recs = interpretation.get('improvement_recommendations', [])
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r.get('suggestion', 'N/A')}")
    print(f"### Confidence in solution: {interpretation.get('confidence_assessment', 'N/A')}")
    print("-" * 60)

def plot_feedback_progress(df, plots_dir):
    """Plots the trajectory of Silhouette scores from Initial -> Feedback."""
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    
    # Filter only sessions that have both Initial and Feedback
    sessions = df.groupby(['Problem', 'Session']).filter(lambda x: len(x) > 1)
    
    if sessions.empty:
        print("No feedback iterations to plot.")
        plt.close()
        return

    # Plot lines connecting Initial to Feedback for each session
    sns.pointplot(
        data=sessions, 
        x='Loop_Stage', 
        y='Silhouette', 
        hue='Problem', 
        markers='o', 
        linestyles='-', 
        dodge=True,
        capsize=0.1
    )
    
    plt.title("Feedback Progress: Improvement per Session")
    plt.ylabel("Silhouette Score")
    plt.xlabel("Execution Stage")
    plt.grid(True, alpha=0.3)
    
    path = plots_dir / "clustering_feedback_progress.png"
    plt.savefig(path)
    plt.close()
    print(f"Feedback progress plot saved to {path}")

def perform_statistical_analysis(df, output_dir, logger):
    print("\n1. WILCOXON SIGNED-RANK TEST: Initial vs Feedback (Per Problem)")
    print("-" * 80)
    
    initial_feedback_stats = {}
    
    for problem in df['Problem'].unique():
        problem_df = df[df['Problem'] == problem]
        initial_scores = problem_df[problem_df['Loop_Stage'] == 'Initial']['Silhouette'].values
        feedback_scores = problem_df[problem_df['Loop_Stage'] == 'Feedback']['Silhouette'].values
        
        # Only compare if both initial and feedback exist
        if len(initial_scores) > 0 and len(feedback_scores) > 0:
            min_len = min(len(initial_scores), len(feedback_scores))
            initial_scores = initial_scores[:min_len]
            feedback_scores = feedback_scores[:min_len]
            
            try:
                from scipy.stats import wilcoxon
                stat, p_value = wilcoxon(initial_scores, feedback_scores)
                
                initial_feedback_stats[problem] = {
                    'initial_mean': np.mean(initial_scores),
                    'feedback_mean': np.mean(feedback_scores),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                sig_marker = "***" if p_value < 0.05 else "ns"
                print(f"{problem:20} | Initial: {np.mean(initial_scores):.4f} -> Feedback: {np.mean(feedback_scores):.4f} | p={p_value:.4f} {sig_marker}")
            except Exception as e:
                logger.error(f"Wilcoxon test error for {problem}: {e}")
    
    # 2. Test between different methods for each problem
    print("\n2. PAIRWISE WILCOXON TEST: Method Comparison (Per Problem)")
    print("-" * 80)
    
    for problem in df['Problem'].unique():
        problem_df = df[df['Problem'] == problem]
        
        if len(problem_df['Method'].unique()) > 1:
            method_data = {}
            for method in problem_df['Method'].unique():
                method_scores = problem_df[problem_df['Method'] == method]['Silhouette'].values
                if len(method_scores) > 0:
                    method_data[method] = method_scores.tolist()
            
            if len(method_data) > 1:
                print(f"\n{problem}:")
                comparison_results = pairwise_wilcoxon_comparison(method_data)
                
                # Print pairwise comparisons
                methods = comparison_results['methods']
                p_values = comparison_results['p_values']
                
                for i in range(len(methods)):
                    for j in range(i + 1, len(methods)):
                        p_val = p_values[i, j]
                        sig = "***" if (not np.isnan(p_val) and p_val < comparison_results['alpha_corrected']) else "ns"
                        print(f"  {methods[i]:15} vs {methods[j]:15} | p={p_val:.4f} {sig}")
    
    # 3. Overall test across all methods
    print("\n3. OVERALL METHOD COMPARISON: All Methods across All Problems")
    print("-" * 80)
    
    all_method_data = {}
    for method in df['Method'].unique():
        method_scores = df[df['Method'] == method]['Silhouette'].values
        if len(method_scores) > 0:
            all_method_data[method] = method_scores.tolist()
    
    if len(all_method_data) > 1:
        comparison_results = pairwise_wilcoxon_comparison(all_method_data)
        print_wilcoxon_summary(comparison_results, "Method Performance Comparison")

    
    # 4. Save statistical results to CSV
    csv_path = output_dir / f"wilcoxon_statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, 'w') as f:
        f.write("Comparison_Type,Problem,Method1,Method2,P_Value,Significant,Sample_Size\n")
        
        for problem, stats in initial_feedback_stats.items():
            sig = "Yes" if stats['significant'] else "No"
            f.write(f"Initial_vs_Feedback,{problem},Initial,Feedback,{stats['p_value']:.6f},{sig},paired\n")
        
        for problem in df['Problem'].unique():
            problem_df = df[df['Problem'] == problem]
            
            if len(problem_df['Method'].unique()) > 1:
                method_data = {}
                for method in problem_df['Method'].unique():
                    method_scores = problem_df[problem_df['Method'] == method]['Silhouette'].values
                    if len(method_scores) > 0:
                        method_data[method] = method_scores.tolist()
                
                if len(method_data) > 1:
                    comparison_results = pairwise_wilcoxon_comparison(method_data)
                    methods = comparison_results['methods']
                    p_values = comparison_results['p_values']
                    significant = comparison_results['significant']
                    
                    for i in range(len(methods)):
                        for j in range(i + 1, len(methods)):
                            p_val = p_values[i, j]
                            sig = "Yes" if significant[i, j] else "No"
                            
                            if not np.isnan(p_val):
                                f.write(f"Method_Comparison,{problem},{methods[i]},{methods[j]},{p_val:.6f},{sig},{len(method_data[methods[i]])}\n")
        
        # Overall method comparison
        if len(all_method_data) > 1:
            comparison_results = pairwise_wilcoxon_comparison(all_method_data)
            methods = comparison_results['methods']
            p_values = comparison_results['p_values']
            significant = comparison_results['significant']
            
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    p_val = p_values[i, j]
                    sig = "Yes" if significant[i, j] else "No"
                    
                    if not np.isnan(p_val):
                        f.write(f"Overall_Method_Comparison,All_Problems,{methods[i]},{methods[j]},{p_val:.6f},{sig},{len(all_method_data[methods[i]])}\n")
    
    logger.info(f"Statistical results CSV saved to {csv_path}")
    print(f"Statistical results CSV saved to: {csv_path}")

def run_clustering_benchmark():
    # Setup Output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = project_root / "outputs" / "results"
    plots_dir = project_root / "outputs" / "figures"
    logs_dir = project_root / "outputs" / "logs"
    memory_dir = project_root / "outputs" / "memory"
    
    for d in [output_dir, plots_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = get_experiment_logger("clustering_benchmark", str(logs_dir))
    logger.info("="*80)
    logger.info("CLUSTERING BENCHMARK STARTED")
    logger.info("="*80)

    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY not found in .env")
        return

    agent = MetaMindAgent(api_key=api_key, verbose=False)
    
    # --- Load Problems ---
    problems = []
    try:
        iris = IrisProblem()
        iris.load_data() 
        problems.append(iris)
    except: pass

    try:
        mall_path = project_root / "data" / "clustering_dataset" / "Mall_Customers.csv"
        if not mall_path.exists(): mall_path = project_root / "data" / "clustering_dataset" / "mall_customers.csv"
        if mall_path.exists():
            mall = MallCustomersProblem()
            mall.load_data(filepath=str(mall_path))
            problems.append(mall)
    except: pass

    try:
        synth = SyntheticClusteringProblem(n_clusters=5)
        synth.load_data(n_samples=500, n_features=5, cluster_std=1.0)
        problems.append(synth)
    except: pass

    all_results = []
    convergence_plots_data = {} # Store history for plotting later

    memory_manager = MemoryManager(output_dir=memory_dir)

    # --- Benchmark Loop ---
    for problem in problems:
        logger.info(f"BENCHMARKING PROBLEM: {problem.problem_name}")
        
        n_sessions = 3
        
        for session_idx in range(n_sessions):
            print(f"\n>>> Session {session_idx+1}/{n_sessions} for {problem.problem_name}")
            
            # 1. Context Construction (Crucial for getting usable params)
            available_methods = {
                'SOM': SOM.PARAM_SPECS,
                'PSO': PSO.PARAM_SPECS,
                'GA': GeneticAlgorithm.PARAM_SPECS,
                'Fuzzy': FuzzyController.PARAM_SPECS
            }
            problem_info = problem.get_info()

            memroy_str = memory_manager.get_context_string(
                problem_type ="clustering",
                problem_name = problem.problem_name,
            )
            
            # [CRITICAL] Hints to guide LLM away from "100 clusters" for 150 samples
            context_hint = "Maximize Silhouette Score."
            if "Iris" in problem.problem_name:
                context_hint += " Dataset is small (150 samples). use VERY SMALL map_size (e.g. 2x2 or 3x3) to force distinct clusters."
            elif "Mall" in problem.problem_name:
                context_hint += " Target 5 distinct segments. Use small map_size (e.g. 3x3)."
            elif "Synthetic" in problem.problem_name:
                context_hint += " CRITICAL: Use map_size around (2,3) to match the 5 true clusters."

            context_hint += f"\n {memroy_str}"
            
            try:
                rec = agent.get_recommendation(
                    problem_info=problem_info,
                    available_methods=available_methods,
                    context=context_hint
                )
            except Exception as e:
                logger.error(f"Agent error: {e}")
                continue
            
            print_llm_json_style(rec)
            
            # 2. Execution Loop
            best_metrics_this_session = {'silhouette': -1.0}
            current_rec = rec
            max_feedback_loops = 1 
            
            for loop_i in range(max_feedback_loops + 1):
                is_feedback_run = loop_i > 0
                run_type = "FEEDBACK RUN" if is_feedback_run else "INITIAL RUN"
                print(f"\n--- {run_type} (Attempt {loop_i+1}) ---")

                MethodClass = get_method_class(current_rec.selected_method)
                
                # Parameter Fixes
                params = current_rec.parameters.copy()
                if 'map_size' in params and isinstance(params['map_size'], list):
                    params['map_size'] = tuple(params['map_size'])
                
                try:
                    method = MethodClass(**params)
                except: break
                
                start_time = time.time()
                try:
                    # Fit
                    fit_data = {'X': problem.X} 
                    if MethodClass == FuzzyController:
                        fit_data.update({'input_range': (np.min(problem.X), np.max(problem.X)), 'output_range': (0,1)})
                        
                    method.fit(fit_data, callback=standard_progress_callback)
                    exec_time = time.time() - start_time
                    
                    # Capture Convergence History
                    if hasattr(method, 'convergence_history') and method.convergence_history:
                        key = f"{problem.problem_name}_{session_idx}_{run_type}"
                        convergence_plots_data[key] = method.convergence_history

                    # Predict
                    labels = None
                    if hasattr(method, 'predict'):
                        labels = method.predict(problem.X)
                        if isinstance(labels, tuple): labels = labels[0]
                    
                    if labels is None: break

                    metrics = evaluate_clustering(problem.X, labels, problem.true_labels)
                    print(f"Result: Silhouette={metrics['silhouette']:.4f} | Clusters={metrics['n_clusters']}")
                    
                    # Store Result
                    result_entry = {
                        'Problem': problem.problem_name,
                        'Session': session_idx + 1,
                        'Loop_Stage': 'Feedback' if is_feedback_run else 'Initial',
                        'Method': current_rec.selected_method,
                        'Parameters': params,
                        'Silhouette': metrics['silhouette'],
                        'ARI': metrics['ari'] if metrics['ari'] else 0.0,
                        'Time': exec_time,
                        'Timestamp': datetime.now().isoformat()
                    }
                    all_results.append(result_entry)

                    if is_feedback_run:
                        memory_manager.save_memory(
                            problem_type = "clustering",
                            entry=result_entry 
                        )

                    # 3. Interpret
                    if loop_i < max_feedback_loops:
                        interpretation = agent.interpret_results(
                            problem_info=problem_info,
                            execution_result={
                                'best_fitness': metrics['silhouette'], 
                                'execution_time': exec_time,
                                'iterations': getattr(method, 'max_epochs', 0),
                                'metrics': metrics
                            },
                            recommendation=current_rec.model_dump()
                        )
                        print_feedback_analysis(interpretation, metrics, best_metrics_this_session['silhouette'])
                        
                        if metrics['silhouette'] < 0.6 or interpretation.get('performance_assessment') == 'poor': 
                            print(f"[Agent] Requesting parameter adjustments...")
                            new_rec = agent.get_feedback_recommendation(
                                problem_info=problem_info,
                                available_methods=available_methods,
                                previous_result={'best_fitness': metrics['silhouette'], 'metrics': metrics},
                                previous_recommendation=current_rec.model_dump()
                            )
                            current_rec = new_rec
                        else:
                            print(f"[Agent] Performance is acceptable. Stopping feedback.")
                            break
                            
                    best_metrics_this_session = metrics

                except Exception as e:
                    logger.error(f"Execution failed: {e}")
                    break

    # -----------------------------------------------------------------------------
    # 3. Final Outputs & Visualization
    # -----------------------------------------------------------------------------
    if all_results:
        # Save Data
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / "clustering_summary.csv", index=False)
        with open(output_dir / f"clustering_results_{timestamp}.json", 'w') as f:
            json.dump(all_results, f, indent=4, default=str)
        
        # 1. Box Plot (Using Utility)
        try:
            box_data = {}
            for (prob, stage), group in df.groupby(['Problem', 'Loop_Stage']):
                box_data[f"{prob}\n({stage})"] = group['Silhouette'].tolist()
            
            plot_box_comparison(
                data_dict=box_data,
                title="Clustering Performance Distribution",
                ylabel="Silhouette Score",
                save_path=str(plots_dir / f"clustering_boxplot_{timestamp}.png"),
                show=False
            )
        except Exception as e: logger.error(f"Boxplot error: {e}")

        # 2. Feedback Progress Plot (New Custom Plot)
        try:
            plot_feedback_progress(df, plots_dir)
        except Exception as e: logger.error(f"Feedback plot error: {e}")

        # 3. Convergence Plots with Confidence Bands (Using Utility)
        try:
            print(f"\nAttempting convergence plots... (convergence_plots_data has {len(convergence_plots_data)} entries)")
            # Plot convergence with bands for each problem
            for prob in [p.problem_name for p in problems]:
                # Collect all convergence histories for this problem
                keys = [k for k in convergence_plots_data.keys() if prob in k]
                print(f"  Problem '{prob}': found {len(keys)} convergence histories")
                if keys:
                    convergence_data = {k: convergence_plots_data[k] for k in keys}
                    if convergence_data:
                        # Group by problem name for the visualization
                        plot_convergence_with_bands(
                            {prob: [convergence_plots_data[k] for k in keys]},
                            title=f"Convergence: {prob} (all sessions with 95% CI)",
                            ylabel="Quantization Error / Fitness",
                            confidence=0.95,
                            save_path=str(plots_dir / f"convergence_bands_{prob.replace(' ', '_')}.png"),
                            show=False
                        )
                        print(f"  Convergence plot saved for {prob}")
            if not convergence_plots_data:
                print("WARNING: convergence_plots_data is empty - no convergence history captured during training")
                logger.warning("No convergence history was captured during training")
        except Exception as e: 
            logger.error(f"Convergence plot error: {e}")
            import traceback
            print(f"ERROR: Convergence plot failed: {e}")
            traceback.print_exc()
        
        # 4. Statistical Analysis - Wilcoxon Tests
        try:
            perform_statistical_analysis(df, output_dir, logger)
        except Exception as e: 
            logger.error(f"Statistical analysis error: {e}")

    logger.info("BENCHMARK COMPLETE")

if __name__ == "__main__":
    run_clustering_benchmark()