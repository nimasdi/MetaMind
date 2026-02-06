import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any

def preprocess_titanic(
    data_dir: Path, 
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Preprocesses the Titanic dataset from 'train.csv'.
    
    Steps:
    1. Load train.csv (ignoring test.csv as it lacks labels for benchmarking).
    2. Missing Values: Age (median), Cabin (feature engineered), Embarked (mode).
    3. Encoding: Sex (binary), Embarked (One-Hot).
    4. Scaling: StandardScaler on numericals.
    5. Split: Train/Val/Test (default 70/15/15).
    """
    file_path = data_dir / "train.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Titanic train.csv not found at {file_path}")

    df = pd.read_csv(file_path)
    
    # --- 1. Handle Missing Values ---
    # Age: Fill with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Embarked: Fill with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Cabin: Engineer into 'HasCabin' (1 if present, 0 else) and drop original
    # This avoids dropping too many rows or dealing with high cardinality
    df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    df = df.drop(columns=['Cabin'])

    # --- 2. Feature Selection & Engineering ---
    # Drop irrelevant IDs and Names
    cols_to_drop = ['PassengerId', 'Name', 'Ticket']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Encode Sex (Binary)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Encode Embarked (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)

    # --- 3. Split Features (X) and Target (y) ---
    target_col = 'Survived'
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # --- 4. Scaling ---
    # We fit scaler only on X (will handle data leakage in split logic roughly here, 
    # but for simplicity in this utility we scale all before splitting or split then scale.
    # Best practice: Split THEN scale.
    
    # First Split: Train (70%) vs Temp (30%)
    train_size = split_ratio[0]
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )

    # Second Split: Val (15%) vs Test (15%) -> relative 0.5 split of Temp
    val_relative = split_ratio[1] / (split_ratio[1] + split_ratio[2])
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_relative, random_state=random_state, stratify=y_temp
    )

    # Scale using ONLY training data statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val,   'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': df.drop(columns=[target_col]).columns.tolist(),
        'scaler': scaler
    }

def preprocess_iris(data_dir: Path) -> Dict[str, Any]:
    """
    Preprocesses the Iris dataset.
    Returns X (scaled features) and y (encoded species).
    """
    file_path = data_dir / "iris.csv" 
    # Handle case where file might be named differently or from sklearn
    if not file_path.exists():
        # Fallback to sklearn if CSV missing, but strictly trying CSV first
        try:
            from sklearn.datasets import load_iris
            iris = load_iris()
            X = iris.data
            y = iris.target
            feature_names = iris.feature_names
            print("Loaded Iris from sklearn (CSV not found).")
        except:
            raise FileNotFoundError(f"Iris dataset not found at {file_path}")
    else:
        df = pd.read_csv(file_path)
        # Assuming standard format: columns 0-3 are features, last is species
        X = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values
        
        # Encode labels if strings
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        feature_names = df.columns[:-1].tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return {
        'X': X_scaled,
        'y': y,
        'feature_names': feature_names,
        'n_clusters': len(np.unique(y))
    }

def preprocess_mall_customers(data_dir: Path) -> Dict[str, Any]:
    """
    Preprocesses the Mall Customers dataset.
    
    Steps:
    1. Drop CustomerID.
    2. Encode Gender.
    3. Scale Age, Income, Score.
    """
    file_path = data_dir / "Mall_Customers.csv"
    if not file_path.exists():
        # Try lowercase
        file_path = data_dir / "mall_customers.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Mall Customers dataset not found in {data_dir}")

    df = pd.read_csv(file_path)

    # Drop non-predictive ID
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    # Rename confusing columns if necessary for consistency
    df.rename(columns={
        'Annual Income (k$)': 'Income',
        'Spending Score (1-100)': 'Score'
    }, inplace=True)

    # Encode Gender
    if 'Gender' in df.columns:
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])

    # Scale all features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    return {
        'X': X,
        'feature_names': df.columns.tolist(),
        'scaler': scaler
    }



# 2026-02-06 19:21:31 - function_optimization - INFO - ================================================================================
# 2026-02-06 19:21:31 - function_optimization - INFO - MetaMind Function Optimization Benchmark
# 2026-02-06 19:21:31 - function_optimization - INFO - ================================================================================
# Initializing MetaMind Agent...
# Agent initialized successfully!


# ================================================================================
# EXPERIMENT CONFIGURATION
# ================================================================================
# Runs per iteration: 5
# Feedback loop: ENABLED ✓
# Max feedback iterations: 2
# ================================================================================

# Memory Manager initialized. Saving to: /Users/nimasaeidi/Desktop/CI_proj/MetaMind/outputs/memory

# ################################################################################
# # Agent-Guided Optimization: Rastrigin-10D
# # Function: Rastrigin, Dimension: 10
# # Optimal Value: 0.0
# # Feedback Loop: ENABLED
# ################################################################################

# ================================================================================
# Asking LLM for recommendation on Rastrigin-10D...
# ================================================================================

# ============================================================
# LLM MULTI-METHOD RECOMMENDATION:
# ============================================================
# {
#     "selected_method": "PSO",
#     "reasoning": "Rastrigin-10D is a continuous, high-dimensional function optimization problem. PSO is well-suited for this type of problem due to its ability to efficiently explore the search space and converge towards the global optimum. The past configurations show that PSO has performed well on this specific problem.",
#     "parameters": {
#         "n_particles": 150,
#         "max_iterations": 1500,
#         "w": 0.7,
#         "c1": 1.5,
#         "c2": 1.5,
#         "w_decay": true,
#         "velocity_clamp": 0.5
#     },
#     "confidence": 0.9,
#     "alternative_methods": ["GA", "DE"],
#     "expected_performance": "high",
#     "warnings": [],
#     "backup_strategy": "If performance is poor, consider increasing the number of particles or iterations, or trying alternative methods like GA or DE with appropriate parameter tuning."
# }
# ============================================================


# LLM Recommendation:
#   Method: PSO
#   Confidence: 90.00%
#   Expected Performance: high
#   Reasoning: Rastrigin-10D is a continuous, high-dimensional function optimization problem. PSO is well-suited for this type of problem due to its ability to efficiently explore the search space and converge towards the global optimum. The past configurations show that PSO has performed well on this specific problem.

#   Recommended Parameters:
#     - n_particles: 150
#     - max_iterations: 1500
#     - w: 0.7
#     - c1: 1.5
#     - c2: 1.5
#     - w_decay: True
#     - velocity_clamp: 0.5

#   Alternative Methods: GA, DE

# ================================================================================
# ITERATION 0: Initial Recommendation
# ================================================================================

# Initial Recommendation - Running 5 independent experiments...
# --------------------------------------------------------------------------------
# [PSO         ] [==============================] 100% | Iter 1500/1500 | Best: 3.9798 Fitness: 3.979836 | Error: 3.979836 | Gap: 397.9836% | Time: 1.81s | Evals: 225150
# [PSO         ] [==============================] 100% | Iter 1500/1500 | Best: 9.9496 Fitness: 9.949586 | Error: 9.949586 | Gap: 994.9586% | Time: 1.81s | Evals: 225150
# [PSO         ] [==============================] 100% | Iter 1500/1500 | Best: 5.9698 Fitness: 5.969754 | Error: 5.969754 | Gap: 596.9754% | Time: 1.81s | Evals: 225150
# [PSO         ] [==============================] 100% | Iter 1500/1500 | Best: 2.9849 Fitness: 2.984877 | Error: 2.984877 | Gap: 298.4877% | Time: 1.81s | Evals: 225150
# [PSO         ] [==============================] 100% | Iter 1500/1500 | Best: 3.9798 Fitness: 3.979836 | Error: 3.979836 | Gap: 397.9836% | Time: 1.81s | Evals: 225150

# ================================================================================
# Iteration 0 Summary
# ================================================================================
# Method: PSO
# Successful Runs: 5/5

# Best Fitness:
#   Best:   2.984877
#   Mean:   5.372778 ± 2.485405
#   Median: 3.979836

# Error from Optimal:
#   Best: 2.984877
#   Mean: 5.372778 ± 2.485405

# Gap Percentage:
#   Best: 298.4877%
#   Mean: 537.2778%

# Computation Time: 1.81s ± 0.00s
# Function Evaluations: 225150
# ================================================================================
# Plot saved to: /Users/nimasaeidi/Desktop/CI_proj/MetaMind/outputs/figures/Rastrigin_10D_iter0_convergence_20260206_192147.png
#   ✓ Saved convergence plot: Rastrigin_10D_iter0_convergence_20260206_192147.png
# Plot saved to: /Users/nimasaeidi/Desktop/CI_proj/MetaMind/outputs/figures/Rastrigin_10D_iter0_all_runs_20260206_192147.png
#   ✓ Saved all-runs plot: Rastrigin_10D_iter0_all_runs_20260206_192147.png
#    [Memory] Saving initial result (Fitness: 2.984877)

# ================================================================================
# STEP 6: LLM Result Interpretation
# ================================================================================

#  Performance Assessment: POOR
# Confidence: LOW

# Analysis:
# The PSO algorithm struggled to find a good solution for the 10D Rastrigin function. With a gap of over 500% from the optimal value, the performance is poor. The convergence was likely erratic and slow, as the algorithm took many iterations (225150) to reach a suboptimal solution. The computation time of 1.81 seconds is reasonable, but the quality of the solution is not satisfactory.

#  Comparison with Expected:
# The actual performance is significantly worse than the expected high performance. The large gap from the optimal value indicates that the PSO algorithm, with the given parameters, is not well-suited for this high-dimensional optimization problem.

#  Improvement Recommendations:
#   1. [PARAMETER_TUNING] Increase the number of particles (n_particles) to improve exploration.
#   2. [PARAMETER_TUNING] Adjust the inertia weight (w) and acceleration coefficients (c1, c2) to balance exploration and exploitation.
#   3. [ALTERNATIVE_METHOD] Try using a Genetic Algorithm (GA) with a larger population size for better exploration.
#   4. [HYBRID_APPROACH] Combine PSO with a local search algorithm, such as the Nelder-Mead method, to refine the solution.

# 🎯 Next Steps:
#   1. Experiment with different parameter settings for PSO.
#   2. Implement and test a Genetic Algorithm with a larger population.
#   3. Develop a hybrid approach combining PSO with a local search algorithm.
#   4. Evaluate the performance of the alternative methods and compare their results.
# ================================================================================

# ================================================================================
# ITERATION 1: Feedback Loop
# ================================================================================

# Requesting feedback from agent...

# Feedback Recommendation:
#   Method: PSO
#   Confidence: 75.00%
#   Reasoning: The gap to optimal is moderate, so we will balance exploration and exploitation. Increasing the swarm size and iterations slightly to allow more search, while keeping other parameters the same to maintain convergence.

#   Adjusted Parameters:
#     🔸 n_particles: 150 → 200
#     🔸 max_iterations: 1500 → 2000
#        w: 0.7 → 0.7
#        c1: 1.5 → 1.5
#        c2: 1.5 → 1.5
#        w_decay: True → True
#        velocity_clamp: 0.5 → 0.5

# Feedback Iteration 1 - Running 5 independent experiments...
# --------------------------------------------------------------------------------
# [PSO         ] [==============================] 100% | Iter 2000/2000 | Best: 5.9697 Fitness: 5.969749 | Error: 5.969749 | Gap: 596.9749% | Time: 3.27s | Evals: 400200
# [PSO         ] [==============================] 100% | Iter 2000/2000 | Best: 2.9849 Fitness: 2.984877 | Error: 2.984877 | Gap: 298.4877% | Time: 3.22s | Evals: 400200
# [PSO         ] [==============================] 100% | Iter 2000/2000 | Best: 3.9798 Fitness: 3.979836 | Error: 3.979836 | Gap: 397.9836% | Time: 3.20s | Evals: 400200
# [PSO         ] [==============================] 100% | Iter 2000/2000 | Best: 3.9798 Fitness: 3.979836 | Error: 3.979836 | Gap: 397.9836% | Time: 3.21s | Evals: 400200
# [PSO         ] [==============================] 100% | Iter 2000/2000 | Best: 5.9698 Fitness: 5.969754 | Error: 5.969754 | Gap: 596.9754% | Time: 3.21s | Evals: 400200

# ================================================================================
# Iteration 1 Summary
# ================================================================================
# Method: PSO
# Successful Runs: 5/5

# Best Fitness:
#   Best:   2.984877
#   Mean:   4.576811 ± 1.193950
#   Median: 3.979836

# Error from Optimal:
#   Best: 2.984877
#   Mean: 4.576811 ± 1.193950

# Gap Percentage:
#   Best: 298.4877%
#   Mean: 457.6811%

# Computation Time: 3.22s ± 0.02s
# Function Evaluations: 400200
# ================================================================================
# Plot saved to: /Users/nimasaeidi/Desktop/CI_proj/MetaMind/outputs/figures/Rastrigin_10D_iter1_convergence_20260206_192217.png
#   ✓ Saved convergence plot: Rastrigin_10D_iter1_convergence_20260206_192217.png
# Plot saved to: /Users/nimasaeidi/Desktop/CI_proj/MetaMind/outputs/figures/Rastrigin_10D_iter1_all_runs_20260206_192217.png
#   ✓ Saved all-runs plot: Rastrigin_10D_iter1_all_runs_20260206_192217.png
#    [Memory] Saving feedback result (Fitness: 2.984877)

# Improvement Analysis:
#   Previous Mean: 5.372778
#   Current Mean:  4.576811
#   Absolute Improvement: 0.795967
#   Percentage Improvement: 14.81%
#   Performance IMPROVED!

# ================================================================================
# ITERATION 2: Feedback Loop
# ================================================================================

# Requesting feedback from agent...

# Feedback Recommendation:
#   Method: PSO
#   Confidence: 70.00%
#   Reasoning: The gap to optimal is moderate, so I will fine-tune the PSO parameters to balance exploration and exploitation. Increasing the swarm size and number of iterations slightly to allow more search. Keeping w_decay enabled to gradually shift from exploration to exploitation.

#   Adjusted Parameters:
#     🔸 n_particles: 200 → 250
#     🔸 max_iterations: 2000 → 2500
#        w: 0.7 → 0.7
#        c1: 1.5 → 1.5
#        c2: 1.5 → 1.5
#        w_decay: True → True
#        velocity_clamp: 0.5 → 0.5
# Failed to create method from feedback: Parameter 'n_particles' must be in range [20, 200], got 250

# ================================================================================
# FINAL STEP 6: LLM Interpretation of Best Results
# ================================================================================

#  Final Assessment: POOR
# Confidence: LOW

#  After 2 iterations of optimization:
# The PSO algorithm struggled to find a good solution for the 10-dimensional Rastrigin function. With a gap of over 450% from the optimal value, the best fitness of 2.98 is far from satisfactory. The convergence was likely erratic and slow, as the algorithm took 400,200 iterations (200 times the specified max_iterations) to reach this suboptimal result. The computation time of 3.22 seconds is reasonable but not impressive given the poor performance.
# ================================================================================

# ================================================================================
# FINAL SUMMARY: Rastrigin-10D
# ================================================================================
# Total Iterations: 2
# Best Iteration: 1
# Best Mean Fitness: 4.576811
# Overall Best Fitness: 2.984877

# Overall Improvement from Initial:
#   Initial Mean: 5.372778
#   Final Mean:   4.576811
#   Total Improvement: 0.795967 (14.81%)
# ================================================================================