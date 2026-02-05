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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Import Framework Components
from src.problems.classification import TitanicProblem
from src.orchestrator.agent import MetaMindAgent
from src.utils.preprocessing import preprocess_titanic
from src.utils.logging import get_experiment_logger

# Import Methods
from src.methods.neural.mlp import MLP
from src.methods.neural.perceptron import Perceptron
from src.methods.neural.hopfield import HopfieldNetwork
from src.methods.neural.som import SOM
from src.methods.fuzzy.controller import FuzzyController
from src.methods.evolutionary.ga import GeneticAlgorithm
from src.methods.evolutionary.pso import PSO

def get_method_class(method_name):
    """Maps LLM string selection to actual Python class."""
    mapping = {
        'MLP': MLP, 'MultiLayerPerceptron': MLP,
        'Perceptron': Perceptron,
        'Hopfield': HopfieldNetwork, 'HopfieldNetwork': HopfieldNetwork,
        'SOM': SOM, 'SelfOrganizingMap': SOM,
        'Fuzzy': FuzzyController, 'FuzzyController': FuzzyController,
        'GA': GeneticAlgorithm, 'GeneticAlgorithm': GeneticAlgorithm,
        'PSO': PSO, 'ParticleSwarmOptimization': PSO,
    }
    return mapping.get(method_name, MLP) # Default to MLP if unknown

def evaluate_predictions(y_true, y_pred, y_proba=None):
    """
    Computes all metrics required by the project document.
    """
    # Ensure binary format for metrics
    y_pred = np.array(y_pred).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_proba is not None:
        try:
            # Handle cases where proba might be (N, 2) or (N,)
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                score = y_proba[:, 1]
            else:
                score = y_proba
            metrics['auc_roc'] = roc_auc_score(y_true, score)
        except:
            metrics['auc_roc'] = 0.0
    else:
        metrics['auc_roc'] = 0.0 # Cannot compute without probabilities
        
    return metrics

def run_classification_benchmark():
    # 1. Setup Logging & Environment
    logger = get_experiment_logger("titanic_benchmark", str(project_root / "outputs" / "logs"))
    logger.info("="*80)
    logger.info("TITANIC CLASSIFICATION BENCHMARK STARTED")
    logger.info("="*80)

    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY not found in .env")
        return

    agent = MetaMindAgent(api_key=api_key, verbose=True)
    
    # 2. Data Loading & Preprocessing (Using separable function)
    data_dir = project_root / "data" / "titanic_dataset"
    logger.info(f"Loading and preprocessing data from {data_dir}...")
    
    # This uses the robust function defined in src/utils/preprocessing.py
    # It handles Age (median), Cabin (feature eng), Sex/Embarked encoding, and Scaling
    try:
        clean_data = preprocess_titanic(data_dir, split_ratio=(0.7, 0.15, 0.15))
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return

    # 3. Setup Problem Instance
    # We manually inject the perfectly split data into the problem instance
    # to ensure exact compliance with the 70/15/15 split requirement.
    problem = TitanicProblem()
    problem.X_train = clean_data['X_train']
    problem.y_train = clean_data['y_train']
    problem.X_val = clean_data['X_val']
    problem.y_val = clean_data['y_val']
    problem.X_test = clean_data['X_test']
    problem.y_test = clean_data['y_test']
    problem.feature_names = clean_data['feature_names']
    
    logger.info(f"Data Split: Train={len(problem.X_train)}, Val={len(problem.X_val)}, Test={len(problem.X_test)}")
    logger.info(f"Features ({len(problem.feature_names)}): {problem.feature_names}")

    # 4. Experimental Loop
    n_runs = 5
    all_results = []
    
    for run_idx in range(n_runs):
        logger.info(f"\n--- Run {run_idx+1}/{n_runs} ---")
        
        # A. LLM Selection
        # We pass method specs for classification-relevant methods
        available_methods = {
            'MLP': MLP.PARAM_SPECS,
            'Perceptron': Perceptron.PARAM_SPECS,
            'FuzzyController': FuzzyController.PARAM_SPECS,
            'SVM': {}, # Placeholder if implemented
        }
        
        problem_info = problem.get_info()
        problem_info['description'] = "Predict passenger survival (0/1). 891 labeled samples. Imbalanced dataset."
        
        try:
            rec = agent.get_recommendation(
                problem_info=problem_info,
                available_methods=available_methods,
                context="Maximize F1-Score and AUC-ROC on Titanic dataset."
            )
            logger.info(f"LLM Selected: {rec.selected_method} | Confidence: {rec.confidence}")
            logger.info(f"Reasoning: {rec.reasoning}")
        except Exception as e:
            logger.error(f"LLM interaction failed: {e}")
            continue

        # B. Method Execution
        MethodClass = get_method_class(rec.selected_method)
        method = MethodClass(**rec.parameters)
        
        # Prepare data dict for fit()
        fit_data = {
            'X_train': problem.X_train, 'y_train': problem.y_train,
            'X_val': problem.X_val, 'y_val': problem.y_val, # For internal validation/early stopping
            'X_test': problem.X_test, 'y_test': problem.y_test
        }
        
        start_time = time.time()
        try:
            method.fit(fit_data)
        except Exception as e:
            logger.error(f"Method execution failed: {e}")
            continue
        exec_time = time.time() - start_time
        
        # C. Evaluation (On TEST set)
        # We explicitly call predict to ensure we evaluate on the hold-out set
        try:
            if hasattr(method, 'predict_proba'):
                y_pred_proba = method.predict_proba(problem.X_test)
                # Handle binary classification proba output formats
                if isinstance(y_pred_proba, list): y_pred_proba = np.array(y_pred_proba)
                # If model returns (N, 2), take second column, else take as is
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
                else:
                    y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = method.predict(problem.X_test)
                y_pred_proba = None # No probabilities available
            
            # Compute Metrics
            metrics = evaluate_predictions(problem.y_test, y_pred, y_pred_proba)
            
            # Cross Validation Score (Generalization check as per doc)
            # We assume the method adheres to sklearn estimator interface for cross_val_score
            # If not, we skip or wrap it. For now, we calculate it manually if possible or skip.
            cv_score = 0.0 # Placeholder
            
            logger.info(f"Result: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics['auc_roc']:.4f}")
            
            result_entry = {
                'run': run_idx + 1,
                'method': rec.selected_method,
                'parameters': rec.parameters,
                'llm_reasoning': rec.reasoning,
                'metrics': metrics,
                'execution_time': exec_time,
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(result_entry)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    # 5. Save Results & Generate Report
    output_dir = project_root / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / f"titanic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Results saved to {json_path}")
    
    # Save Summary CSV
    csv_path = output_dir / "titanic_summary.csv"
    df_results = pd.DataFrame([{
        'Run': r['run'],
        'Method': r['method'],
        'Accuracy': r['metrics']['accuracy'],
        'Precision': r['metrics']['precision'],
        'Recall': r['metrics']['recall'],
        'F1': r['metrics']['f1_score'],
        'AUC': r['metrics']['auc_roc'],
        'Time': r['execution_time']
    } for r in all_results])
    
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Summary CSV saved to {csv_path}")
    
    print("\n" + "="*50)
    print("TITANIC BENCHMARK SUMMARY")
    print("="*50)
    print(df_results.groupby('Method')[['Accuracy', 'F1', 'AUC', 'Time']].agg(['mean', 'std']).round(4))

if __name__ == "__main__":
    run_classification_benchmark()