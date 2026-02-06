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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# -----------------------------------------------------------------------------
# 0. Setup Path & Environment
# -----------------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Import Framework Components
from src.problems.classification import TitanicProblem
from src.orchestrator.agent import MetaMindAgent
from src.utils.preprocessing import preprocess_titanic

# Import Utilities
from src.utils.logging import get_experiment_logger, standard_progress_callback
from src.utils.plotting import plot_box_comparison, plot_convergence, plot_convergence_with_bands
from src.utils.metrics import pairwise_wilcoxon_comparison, print_wilcoxon_summary

# Import Methods
from src.methods.neural.mlp import MLP
from src.methods.neural.perceptron import Perceptron
from src.methods.neural.hopfield import HopfieldNetwork
from src.methods.neural.som import SOM
from src.methods.fuzzy.controller import FuzzyController
from src.methods.evolutionary.ga import GeneticAlgorithm
from src.methods.evolutionary.pso import PSO
from src.orchestrator.memory import MemoryManager

def get_method_class(method_name):
    """Maps LLM string selection to actual Python class."""
    mapping = {
        'MLP': MLP, 'MultiLayerPerceptron': MLP,
        'Perceptron': Perceptron,
        'Fuzzy': FuzzyController, 'FuzzyController': FuzzyController,
        # Less likely for classification but available
        'GA': GeneticAlgorithm, 'GeneticAlgorithm': GeneticAlgorithm,
        'PSO': PSO, 'ParticleSwarmOptimization': PSO,
    }
    return mapping.get(method_name, MLP)

def evaluate_predictions(y_true, y_pred, y_proba=None):
    """Computes classification metrics."""
    y_pred = np.array(y_pred).astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_proba is not None:
        try:
            # Handle multi-class or binary proba shapes
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                score = y_proba[:, 1]
            else:
                score = y_proba.ravel()
            metrics['auc_roc'] = float(roc_auc_score(y_true, score))
        except:
            metrics['auc_roc'] = 0.0
    else:
        metrics['auc_roc'] = 0.0
        
    return metrics

def print_llm_json_style(rec):
    """Prints the LLM recommendation in the requested Document format."""
    output = {
        "problem_type": "classification",
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
    """Prints the feedback analysis in the requested Document format."""
    print("\nLLM feedback output:")
    print("## Results Analysis")
    
    score = metrics.get('f1_score', 0)
    imp_str = ""
    if previous_best != -1.0:
        diff = score - previous_best
        imp_str = f"(Change: {diff:+.4f})"

    print(f"The method achieved an F1-Score of {score:.4f} {imp_str}.")
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
    """Plots the trajectory of F1-Scores from Initial -> Feedback."""
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    
    # Filter only sessions that have both Initial and Feedback
    sessions = df.groupby(['Problem', 'Session']).filter(lambda x: len(x) > 1)
    
    if sessions.empty:
        return

    # Plot lines connecting Initial to Feedback for each session
    sns.pointplot(
        data=sessions, 
        x='Loop_Stage', 
        y='F1_Score', 
        hue='Session', 
        markers='o', 
        linestyles='-', 
        dodge=True,
        capsize=0.1,
        palette='viridis'
    )
    
    plt.title("Feedback Progress: F1-Score Improvement per Session")
    plt.ylabel("F1-Score (Higher is better)")
    plt.xlabel("Execution Stage")
    plt.grid(True, alpha=0.3)
    
    path = plots_dir / "classification_feedback_progress.png"
    plt.savefig(path)
    plt.close()

def perform_statistical_analysis(df, output_dir, logger):
    """Perform Wilcoxon statistical tests on classification results."""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - WILCOXON TEST RESULTS")
    print("="*80)
    
    # 1. Test Initial vs Feedback
    print("\n1. WILCOXON SIGNED-RANK TEST: Initial vs Feedback")
    print("-" * 80)
    
    initial_feedback_stats = {}
    
    initial_scores = df[df['Loop_Stage'] == 'Initial']['F1_Score'].values
    feedback_scores = df[df['Loop_Stage'] == 'Feedback']['F1_Score'].values
    
    if len(initial_scores) > 0 and len(feedback_scores) > 0:
        min_len = min(len(initial_scores), len(feedback_scores))
        initial_scores = initial_scores[:min_len]
        feedback_scores = feedback_scores[:min_len]
        
        try:
            from scipy.stats import wilcoxon
            stat, p_value = wilcoxon(initial_scores, feedback_scores)
            
            initial_feedback_stats['Titanic'] = {
                'initial_mean': np.mean(initial_scores),
                'feedback_mean': np.mean(feedback_scores),
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            sig_marker = "***" if p_value < 0.05 else "ns"
            print(f"Titanic: Initial: {np.mean(initial_scores):.4f} -> Feedback: {np.mean(feedback_scores):.4f} | p={p_value:.4f} {sig_marker}")
        except Exception as e:
            logger.error(f"Wilcoxon test error: {e}")
    
    # 2. Test between different methods
    print("\n2. PAIRWISE WILCOXON TEST: Method Comparison")
    print("-" * 80)
    
    method_data = {}
    for method in df['Method'].unique():
        method_scores = df[df['Method'] == method]['F1_Score'].values
        if len(method_scores) > 0:
            method_data[method] = method_scores.tolist()
    
    if len(method_data) > 1:
        comparison_results = pairwise_wilcoxon_comparison(method_data)
        
        methods = comparison_results['methods']
        p_values = comparison_results['p_values']
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                p_val = p_values[i, j]
                sig = "***" if (not np.isnan(p_val) and p_val < comparison_results['alpha_corrected']) else "ns"
                print(f"{methods[i]:15} vs {methods[j]:15} | p={p_val:.4f} {sig}")
        
        print_wilcoxon_summary(comparison_results, "Method Performance Comparison")
    
    # 3. Save statistical summary
    stats_file = output_dir / "wilcoxon_statistical_analysis.txt"
    with open(stats_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WILCOXON STATISTICAL TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. INITIAL vs FEEDBACK (Paired Wilcoxon Signed-Rank Test)\n")
        f.write("-" * 80 + "\n")
        for problem, stats in initial_feedback_stats.items():
            sig = "SIGNIFICANT (p<0.05)" if stats['significant'] else "NOT SIGNIFICANT"
            f.write(f"{problem}:\n")
            f.write(f"  Initial Mean F1-Score:  {stats['initial_mean']:.4f}\n")
            f.write(f"  Feedback Mean F1-Score: {stats['feedback_mean']:.4f}\n")
            f.write(f"  P-value: {stats['p_value']:.4f} ({sig})\n\n")
        
        f.write("\n2. METHOD COMPARISON (Bonferroni-Corrected Pairwise Wilcoxon Tests)\n")
        f.write("-" * 80 + "\n")
        if len(method_data) > 1:
            comparison_results = pairwise_wilcoxon_comparison(method_data)
            f.write(f"Bonferroni-corrected significance level: {comparison_results['alpha_corrected']:.6f}\n\n")
            
            methods = comparison_results['methods']
            p_values = comparison_results['p_values']
            
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    p_val = p_values[i, j]
                    sig = comparison_results['significant'][i, j]
                    f.write(f"{methods[i]} vs {methods[j]}: p={p_val:.4f} {'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'}\n")
    
    logger.info(f"Statistical analysis saved to {stats_file}")
    print(f"\nStatistical analysis saved to: {stats_file}")
    
    # 4. Save to CSV
    csv_path = output_dir / f"wilcoxon_statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, 'w') as f:
        f.write("Comparison_Type,Method1,Method2,P_Value,Significant,Sample_Size\n")
        
        # Initial vs Feedback
        for problem, stats in initial_feedback_stats.items():
            sig = "Yes" if stats['significant'] else "No"
            f.write(f"Initial_vs_Feedback,Initial,Feedback,{stats['p_value']:.6f},{sig},paired\n")
        
        # Method comparisons
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
                        f.write(f"Method_Comparison,{methods[i]},{methods[j]},{p_val:.6f},{sig},{len(method_data[methods[i]])}\n")
    
    logger.info(f"Statistical results CSV saved to {csv_path}")
    print(f"Statistical results CSV saved to: {csv_path}")

def run_classification_benchmark():

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = project_root / "outputs" / "results"
    plots_dir = project_root / "outputs" / "figures"
    logs_dir = project_root / "outputs" / "logs"
    memory_dir = project_root / "outputs" / "memory"
    
    for d in [output_dir, plots_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Logging
    logger = get_experiment_logger("classification_benchmark", str(logs_dir))
    logger.info("="*80)
    logger.info("TITANIC CLASSIFICATION BENCHMARK STARTED (Agent Loop Enabled)")
    logger.info("="*80)

    # API Key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY not found in .env")
        return

    agent = MetaMindAgent(api_key=api_key, verbose=False)
    
    # --- 1. Data Loading & Preprocessing ---
    data_dir = project_root / "data" / "titanic_dataset"
    logger.info(f"Loading data from {data_dir}...")
    
    try:
        clean_data = preprocess_titanic(data_dir, split_ratio=(0.7, 0.15, 0.15))
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return

    # Setup Problem Object
    problem = TitanicProblem()
    problem.X_train = clean_data['X_train']
    problem.y_train = clean_data['y_train']
    problem.X_val = clean_data['X_val']
    problem.y_val = clean_data['y_val']
    problem.X_test = clean_data['X_test']
    problem.y_test = clean_data['y_test']
    problem.feature_names = clean_data['feature_names']
    
    logger.info(f"Data Split: Train={len(problem.X_train)}, Val={len(problem.X_val)}, Test={len(problem.X_test)}")

    all_results = []
    convergence_plots_data = {}

    # --- 2. Benchmark Loop ---
    n_sessions = 3

    memory_manager = MemoryManager(output_dir=memory_dir)
    
    for session_idx in range(n_sessions):
        print(f"\n>>> Session {session_idx+1}/{n_sessions} for Titanic Classification")
        
        # Step 1: Initial Recommendation
        # We only expose valid classification methods to the LLM
        available_methods = {
            'MLP': MLP.PARAM_SPECS,
            'Perceptron': Perceptron.PARAM_SPECS,
            'FuzzyController': FuzzyController.PARAM_SPECS,
        }
        
        problem_info = problem.get_info()
        problem_info['description'] = "Predict passenger survival (0/1). Imbalanced dataset (approx 60/40)."

        # memory loading
        memory_str = memory_manager.get_context_string(
            problem_type="classification",
            problem_name="Titanic",
        )
        
        context_hint = (
            "Maximize F1-Score and AUC-ROC. on the Test set. "
            "Data is imbalanced, so Accuracy alone is misleading. "
            "Prefer MLP for non-linear patterns or Perceptron for simple baselines."
            "CRITICAL WARNING: The dataset is VERY SMALL (<1000 samples). "
            "1. Do NOT use large architectures. Keep hidden_layers small (e.g., [64, 32] or [32]). "
            "2. If performing Feedback: Do NOT add more layers. Instead, reduce learning_rate or increase regularization (dropout/weight decay). "
            "3. Large models will overfit and lower the Test F1-Score."
            f"{memory_str}"
        )
        
        try:
            rec = agent.get_recommendation(
                problem_info=problem_info,
                available_methods=available_methods,
                context=context_hint
            )
        except Exception as e:
            logger.error(f"Agent recommendation failed: {e}")
            continue
        
        print_llm_json_style(rec)
        
        # Step 2: Execution & Feedback Loop
        best_metrics_this_session = {'f1_score': -1.0}
        current_rec = rec
        max_feedback_loops = 1 
        
        for loop_i in range(max_feedback_loops + 1):
            is_feedback_run = loop_i > 0
            run_type = "FEEDBACK RUN" if is_feedback_run else "INITIAL RUN"
            print(f"\n--- {run_type} (Attempt {loop_i+1}) ---")

            # Instantiate
            MethodClass = get_method_class(current_rec.selected_method)
            params = current_rec.parameters.copy()
            
            # Clean params if lists are passed where tuples expected (usually not issue for MLP/Perceptron)
            try:
                method = MethodClass(**params)
            except Exception as e:
                logger.error(f"Instantiation failed: {e}")
                break
            
            # Fit
            fit_data = {
                'X_train': problem.X_train, 'y_train': problem.y_train,
                'X_val': problem.X_val, 'y_val': problem.y_val, 
                'X_test': problem.X_test, 'y_test': problem.y_test
            }
            
            # Special handling for Fuzzy Controller ranges
            if MethodClass == FuzzyController:
                fit_data['input_range'] = (np.min(problem.X_train), np.max(problem.X_train))
                fit_data['output_range'] = (0, 1)
                fit_data['input_data'] = problem.X_train # Required for Wang-Mendel
                fit_data['output_data'] = problem.y_train

            start_time = time.time()
            try:
                method.fit(fit_data, callback=standard_progress_callback)
                exec_time = time.time() - start_time
                
                # Capture Convergence
                if hasattr(method, 'convergence_history') and method.convergence_history:
                    key = f"Titanic_{session_idx}_{run_type}_{current_rec.selected_method}"
                    convergence_plots_data[key] = method.convergence_history

                # Evaluate (on TEST set for final report)
                y_pred_proba = None
                if hasattr(method, 'predict_proba'):
                    try:
                        y_pred_proba = method.predict_proba(problem.X_test)
                    except: pass
                
                y_pred = method.predict(problem.X_test)
                
                metrics = evaluate_predictions(problem.y_test, y_pred, y_pred_proba)
                
                print(f"Result: Accuracy={metrics['accuracy']:.4f} | F1={metrics['f1_score']:.4f} | AUC={metrics['auc_roc']:.4f}")
                
                # Store Result
                result_entry = {
                    'Problem': 'Titanic',
                    'Session': session_idx + 1,
                    'Loop_Stage': 'Feedback' if is_feedback_run else 'Initial',
                    'Method': current_rec.selected_method,
                    'Parameters': params,
                    'Accuracy': metrics['accuracy'],
                    'F1_Score': metrics['f1_score'],
                    'AUC': metrics['auc_roc'],
                    'Recall': metrics['recall'],
                    'Time': exec_time,
                    'Timestamp': datetime.now().isoformat()
                }
                all_results.append(result_entry)

                if is_feedback_run:
                    memory_manager.save_memory(
                        problem_type = "classification",
                        entry=result_entry
                    )

                # Step 3: Interpret & Feedback
                if loop_i < max_feedback_loops:
                    interpretation = agent.interpret_results(
                        problem_info=problem_info,
                        execution_result={
                            'best_fitness': metrics['f1_score'], # Treating F1 as fitness
                            'execution_time': exec_time,
                            'iterations': getattr(method, 'max_epochs', 0),
                            'metrics': metrics
                        },
                        recommendation=current_rec.model_dump()
                    )
                    
                    print_feedback_analysis(interpretation, metrics, best_metrics_this_session['f1_score'])
                    
                    # Trigger feedback if F1 is poor (< 0.75) or explicitly suggested
                    if metrics['f1_score'] < 0.78 or interpretation.get('performance_assessment') == 'poor': 
                        print(f"[Agent] Requesting parameter adjustments to boost F1-Score...")
                        new_rec = agent.get_feedback_recommendation(
                            problem_info=problem_info,
                            available_methods=available_methods,
                            previous_result={'best_fitness': metrics['f1_score'], 'metrics': metrics},
                            previous_recommendation=current_rec.model_dump()
                        )
                        current_rec = new_rec
                    else:
                        print(f"[Agent] Performance is good (F1 > 0.78). Stopping feedback loop.")
                        break
                        
                best_metrics_this_session = metrics

            except Exception as e:
                logger.error(f"Run execution failed: {e}")
                import traceback
                traceback.print_exc()
                break

    # -----------------------------------------------------------------------------
    # 3. Outputs & Visualization
    # -----------------------------------------------------------------------------
    if all_results:
        with open(output_dir / f"classification_results_{timestamp}.json", 'w') as f:
            json.dump(all_results, f, indent=4, default=str)
        logger.info(f"Results saved to JSON")
        
        df = pd.DataFrame(all_results)
        csv_path = output_dir / "classification_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV saved")
        
        print("\n" + "="*60)
        print("TITANIC BENCHMARK SUMMARY (Aggregated)")
        print("="*60)
        summary = df.groupby(['Method', 'Loop_Stage'])[['Accuracy', 'F1_Score', 'AUC', 'Time']].mean().round(4)
        print(summary)
        
        print("\nGenerating plots...")
        
        # 1. Box Plot (F1 Score)
        try:
            box_data = {}
            for (method, stage), group in df.groupby(['Method', 'Loop_Stage']):
                label = f"{method}\n({stage})"
                box_data[label] = group['F1_Score'].tolist()
            
            plot_box_comparison(
                data_dict=box_data,
                title="Classification Performance (F1-Score)",
                ylabel="F1-Score",
                save_path=str(plots_dir / f"classification_boxplot_{timestamp}.png"),
                show=False
            )
        except Exception as e: logger.error(f"Boxplot error: {e}")

        # 2. Feedback Progress Plot
        try:
            plot_feedback_progress(df, plots_dir)
        except Exception as e: logger.error(f"Feedback plot error: {e}")

        # 3. Convergence Plots with Confidence Bands
        try:
            if convergence_plots_data:
                convergence_by_method = {}
                for key, history in convergence_plots_data.items():
                    parts = key.split('_')
                    method_name = parts[-1] if len(parts) > 0 else "Unknown"
                    
                    if method_name not in convergence_by_method:
                        convergence_by_method[method_name] = []
                    convergence_by_method[method_name].append(history)
                
                print(f"Convergence histories organized by method: {list(convergence_by_method.keys())}")
                
                plot_convergence_with_bands(
                    convergence_by_method,
                    title="Classification - Mean Convergence with 95% Confidence Bands",
                    xlabel="Epoch",
                    ylabel="Training Loss/Metric",
                    confidence=0.95,
                    save_path=str(plots_dir / f"classification_convergence_bands_{timestamp}.png"),
                    show=False
                )
            else:
                print(f"WARNING: convergence_plots_data is empty - no convergence history captured during training")
                logger.warning("No convergence history was captured during training")
        except Exception as e: 
            logger.error(f"Convergence with bands plot error: {e}")
            import traceback
            print(f"ERROR: Convergence plot failed: {e}")
            traceback.print_exc()
        
        try:
            perform_statistical_analysis(df, output_dir, logger)
        except Exception as e: 
            logger.error(f"Statistical analysis error: {e}")

    logger.info("BENCHMARK COMPLETE")

if __name__ == "__main__":
    run_classification_benchmark()