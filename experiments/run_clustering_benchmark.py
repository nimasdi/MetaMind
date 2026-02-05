import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.decomposition import PCA

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Import Framework Components
from src.problems.clustering import ClusteringProblem, IrisProblem, MallCustomersProblem, SyntheticClusteringProblem
from src.orchestrator.agent import MetaMindAgent
from src.utils.logging import get_experiment_logger

# Import Methods
from src.methods.neural.som import SOM
from src.methods.evolutionary.ga import GeneticAlgorithm
from src.methods.evolutionary.pso import PSO
# Assuming a generic wrapper or using the methods directly for clustering
# Note: GA/PSO for clustering usually optimize centroids. 
# If specific clustering implementations aren't in the methods, SOM is the primary CI clustering tool.

def get_method_class(method_name):
    """Maps LLM string selection to actual Python class."""
    mapping = {
        'SOM': SOM, 'SelfOrganizingMap': SOM, 'Kohonen': SOM,
        'GA': GeneticAlgorithm, 'GeneticAlgorithm': GeneticAlgorithm,
        'PSO': PSO, 'ParticleSwarmOptimization': PSO,
        # Add Fuzzy C-Means if available in src/methods/fuzzy
    }
    # Default to SOM for clustering if unknown, as it's the standard CI method
    return mapping.get(method_name, SOM) 

def evaluate_clustering(X, labels, true_labels=None):
    """
    Computes clustering metrics as per project requirements.
    """
    n_labels = len(np.unique(labels))
    
    # metrics require > 1 cluster and < n_samples
    if n_labels < 2 or n_labels >= len(X):
        return {
            'silhouette': -1.0, 'davies_bouldin': float('inf'), 'calinski_harabasz': 0.0,
            'ari': 0.0, 'nmi': 0.0, 'inertia': 0.0, 'n_clusters': n_labels
        }

    metrics = {
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'n_clusters': n_labels
    }
    
    # Metrics requiring ground truth
    if true_labels is not None:
        metrics['ari'] = adjusted_rand_score(true_labels, labels)
        metrics['nmi'] = normalized_mutual_info_score(true_labels, labels)
    else:
        metrics['ari'] = None
        metrics['nmi'] = None
        
    return metrics

def run_clustering_benchmark():
    # 1. Setup Logging & Environment
    logger = get_experiment_logger("clustering_benchmark", str(project_root / "outputs" / "logs"))
    logger.info("="*80)
    logger.info("CLUSTERING BENCHMARK STARTED")
    logger.info("="*80)

    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY not found in .env")
        return

    agent = MetaMindAgent(api_key=api_key, verbose=True)
    
    # 2. Define Problems
    problems = []
    
    # A. Iris (Validation)
    try:
        # Check specific path first
        iris_path = project_root / "data" / "clustering_dataset" / "Iris.csv"
        
        iris = IrisProblem()
        if iris_path.exists():
             # Assuming load_data can take a filepath if implemented to do so, 
             # otherwise rely on internal logic but prefer our path if it matches standard format
             # For standard IrisProblem, we often rely on sklearn, but let's try to use the file if present
             iris.load_data(filepath=str(iris_path))
             logger.info(f"[+] Loaded Dataset A: Iris from {iris_path}")
        else:
             # Fallback to internal sklearn load
             iris.load_data() 
             logger.info("[+] Loaded Dataset A: Iris (sklearn fallback)")
             
        problems.append(iris)
        
    except Exception as e:
        logger.error(f"Failed to load Iris: {e}")

    # B. Mall Customers
    try:
        mall_path = project_root / "data" / "clustering_dataset" / "Mall_Customers.csv"
        # If file is lowercase
        if not mall_path.exists():
             mall_path = project_root / "data" / "clustering_dataset" / "mall_customers.csv"
             
        if mall_path.exists():
            mall = MallCustomersProblem()
            mall.load_data(filepath=str(mall_path))
            problems.append(mall)
            logger.info(f"[+] Loaded Dataset B: Mall Customers from {mall_path}")
        else:
            logger.warning(f"⚠ Mall Customers CSV not found at {mall_path}")
    except Exception as e:
        logger.error(f"Failed to load Mall Customers: {e}")

    # C. Synthetic Data (Controlled)
    try:
        synth = SyntheticClusteringProblem(n_clusters=5)
        synth.load_data(n_samples=500, n_features=5, cluster_std=1.0)
        problems.append(synth)
        logger.info("[+] Generated Dataset C: Synthetic (500 samples, 5 features)")
    except Exception as e:
        logger.error(f"Failed to generate Synthetic data: {e}")

    # 3. Experimental Loop
    n_runs = 5
    all_results = []
    
    # Cache for API limit handling
    recommendation_cache = {} 

    for problem in problems:
        logger.info(f"\n{'='*60}")
        logger.info(f"BENCHMARKING PROBLEM: {problem.problem_name}")
        logger.info(f"{'='*60}")
        
        last_successful_rec = None
        
        for run_idx in range(n_runs):
            logger.info(f"\n--- Run {run_idx+1}/{n_runs} for {problem.problem_name} ---")
            
            # --- A. LLM Selection ---
            # Define available methods for clustering
            available_methods = {
                'SOM': SOM.PARAM_SPECS,
                # Include others if your GA/PSO classes have specific clustering modes
                # Otherwise, SOM is the main one expected by the doc
            }
            
            problem_info = problem.get_info()
            
            rec = None
            try:
                # Check if we already have a cached decision for this problem from previous runs
                # (Optional optimization, but strictly we should ask every time unless rate limited)
                rec = agent.get_recommendation(
                    problem_info=problem_info,
                    available_methods=available_methods,
                    context="Select the best clustering method. For Mall Customers use 4-6 clusters. Maximize Silhouette Score."
                )
                last_successful_rec = rec
                recommendation_cache[problem.problem_name] = rec
                logger.info(f"LLM Selected: {rec.selected_method} | Confidence: {rec.confidence}")
                
            except Exception as e:
                logger.error(f"LLM interaction failed: {e}")
                if last_successful_rec:
                    logger.warning("⚠️ API Limit Reached. Reusing cached recommendation.")
                    rec = last_successful_rec
                elif problem.problem_name in recommendation_cache:
                    logger.warning("⚠️ API Limit Reached. Reusing recommendation from previous problem run.")
                    rec = recommendation_cache[problem.problem_name]
                else:
                    # Hard fallback if LLM completely fails
                    from src.core.types import LLMRecommendation
                    logger.warning("⚠️ No cache available. Using Hardcoded Default (SOM).")
                    rec = LLMRecommendation(
                        selected_method="SOM",
                        reasoning="Fallback due to API error",
                        parameters=SOM.get_default_parameters(),
                        confidence=0.0,
                        alternative_methods=[],
                        expected_performance="medium",
                        warnings=["Using default fallback"]
                    )

            # --- B. Execution ---
            MethodClass = get_method_class(rec.selected_method)
            
            # Fix parameter types (JSON returns lists for tuples)
            current_params = rec.parameters.copy()
            if 'map_size' in current_params and isinstance(current_params['map_size'], list):
                current_params['map_size'] = tuple(current_params['map_size'])
                
            # Ensure parameters are valid for the class
            method = MethodClass(**current_params)
            
            start_time = time.time()
            try:
                # Clustering fit usually takes just X
                # But BaseMethod.fit expects a dict 'problem_data'
                fit_data = {'X': problem.X}
                if hasattr(problem, 'y') or hasattr(problem, 'true_labels'):
                     # Pass true labels if available just in case method uses them for something (unlikely for unsupervised)
                     fit_data['y'] = problem.true_labels 
                
                method.fit(fit_data)
            except Exception as e:
                logger.error(f"Method execution failed: {e}")
                import traceback
                traceback.print_exc()
                continue
                
            exec_time = time.time() - start_time
            
            # --- C. Evaluation ---
            try:
                results = method.get_results()
                
                # Extract labels (SOM might return them differently, standardized check)
                labels = None
                if 'labels' in results:
                    labels = results['labels']
                elif 'best_solution' in results:
                    # If best_solution contains centroids, we need to predict labels
                    # If best_solution contains labels, use them
                    bs = results['best_solution']
                    if hasattr(bs, 'shape') and bs.shape[0] == problem.X.shape[0]:
                        labels = bs
                    elif hasattr(method, 'predict'):
                        labels = method.predict(problem.X)
                
                if labels is None and hasattr(method, 'predict'):
                     labels = method.predict(problem.X)
                     
                if labels is not None:
                    metrics = evaluate_clustering(problem.X, labels, problem.true_labels)
                    
                    logger.info(f"Silhouette: {metrics['silhouette']:.4f}")
                    if metrics['ari'] is not None:
                        logger.info(f"ARI: {metrics['ari']:.4f}")
                        
                    result_entry = {
                        'problem': problem.problem_name,
                        'run': run_idx + 1,
                        'method': rec.selected_method,
                        'parameters': rec.parameters,
                        'metrics': metrics,
                        'execution_time': exec_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    all_results.append(result_entry)
                else:
                    logger.error("Could not extract cluster labels from method.")

            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                import traceback
                traceback.print_exc()

    # 4. Save Results & Plotting
    output_dir = project_root / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = project_root / "outputs" / "figures"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Results saved to {json_path}")
    
    # Save CSV
    if all_results:
        df_results = pd.DataFrame([{
            'Problem': r['problem'],
            'Method': r['method'],
            'Run': r['run'],
            'Silhouette': r['metrics']['silhouette'],
            'DB_Index': r['metrics']['davies_bouldin'],
            'CH_Index': r['metrics']['calinski_harabasz'],
            'ARI': r['metrics']['ari'],
            'Time': r['execution_time']
        } for r in all_results])
        
        csv_path = output_dir / "clustering_summary.csv"
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV saved to {csv_path}")
        
        # Plot: Silhouette Scores Box Plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_results, x='Problem', y='Silhouette', hue='Method')
        plt.title("Clustering Performance (Silhouette Score)")
        plt.tight_layout()
        plt.savefig(plots_dir / "clustering_silhouette_comparison.png")
        logger.info(f"Plot saved to {plots_dir}")

        # Summary Print
        print("\n" + "="*60)
        print("CLUSTERING BENCHMARK SUMMARY")
        print("="*60)
        print(df_results.groupby(['Problem', 'Method'])[['Silhouette', 'DB_Index', 'ARI']].agg(['mean', 'std']).round(4))

if __name__ == "__main__":
    run_clustering_benchmark()