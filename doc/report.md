## Section 1: Intro

### 1.1 Project Objectives

The primary objective of this project is to design and implement **MetaMind**, an intelligent computational framework that leverages Large Language Models (LLMs) to automate the selection, configuration, and execution of Computational Intelligence (CI) methods. Traditional approaches to solving complex problems, whether optimization, classification, or clustering, often rely heavily on expert intuition and manual trial-and-error to select the appropriate algorithm and tune its hyperparameters.

MetaMind aims to bridge this gap by acting as an Intelligent Architect. By integrating a semantic understanding of problem descriptions with a rigorous execution engine, the system achieves the following specific goals:

1. **Adaptive Model Selection:** To autonomously analyze problem metadata (e.g., dimensionality, data type, constraints) and select the most suitable algorithm from a diverse library of 9 implemented CI methods.

2. **Dynamic Parameter Tuning:** To utilize the inferential capabilities of LLMs to suggest optimal initial hyperparameters like mutation rates, learning rates, network topology and ... tailored to the specific instance.

3. **Iterative Feedback Loop:** To implement a closed-loop optimization process where the LLM interprets execution results and provides corrective feedback to refine parameters for subsequent runs, thereby minimizing the performance gap.

4. **Comprehensive Benchmarking:** To validate the framework's efficacy across four distinct problem domains using standard benchmarks, ensuring both versatility and robustness.


### 1.2 System Overview

The MetaMind system is built upon a modular architecture centered around the `Orchestrator` class, which serves as the bridge between the cognitive agent and the computational workers. The high-level workflow is as follows:

- **Problem Definition:** The user instantiates a problem object (e.g., `TSPProblem`, `TitanicProblem`), which encapsulates data, objective functions, and constraints.

- **The MetaMind Agent:** This cognitive core utilizes the `Llama-3.3-70b-versatile` (or at the time of presentation something else cause of the constant rate limits we're facing right now) model via the **Groq API**(also maybe something else at the time of presentation). It receives a structured prompt containing problem metadata and the `PARAM_SPECS` of available methods. Using **JSON Mode**, it outputs a structured recommendation containing the selected method, reasoning, and parameters.

- **Execution Pipeline:** The `Orchestrator` dynamically instantiates the recommended method (e.g., `GeneticAlgorithm` or `MLP`) and executes the `fit()` method on the prepared data.

- **Feedback Mechanism:** If the performance gap exceeds a defined threshold (e.g., 5%), the system enters a feedback loop. Execution metrics are fed back to the LLM, which analyzes convergence behavior and suggests specific parameter adjustments to improve the solution.

+ **Structured Cognitive Processing:** Unlike varying text-based approaches, This structure is going to enforces a strict **JSON Schema** validation for all LLM outputs. This ensures that the agent's reasoning is machine-parsable and that parameters (e.g., `learning_rate`, `mutation_rate`) are strictly typed and bounded before execution, preventing runtime type errors common in LLM integrations.

The workflow is something like the below sequence, you're gonna see a visual form of it in other kind in the next part too:

```
graph TD
    User[User / Problem Definition] -->|Input Data| Orch[Orchestrator]
    Orch -->|1. Request Strategy| Agent[MetaMind Agent (LLM)]
    Agent -->|2. JSON Recommendation| Orch
    Orch -->|3. Instantiate & Fit| Method[CI Method (e.g., GA, MLP)]
    Method -->|4. Execution Results| Orch
    Orch -->|5. Check Performance Gap| Eval{Gap < 5%?}
    Eval -- Yes --> Final[Final Report]
    Eval -- No --> Feedback[Feedback Loop]
    Feedback -->|6. Result Interpretation| Agent
    Agent -->|7. Adjusted Parameters| Orch
```
### 1.3 Methods and Problems Summary

**A. Method Library**

The framework incorporates 9 distinct Computational Intelligence methods across four categories:

1. **Evolutionary Algorithms:**
    
    - **Genetic Algorithm (GA):** Implemented using `DEAP`, featuring configurable selection (tournament/roulette) and crossover (PMX/CX) operators.
    
    - **Genetic Programming (GP):** Tree-based evolution for symbolic regression tasks.
    
2. **Swarm Intelligence:**
    
    - **Particle Swarm Optimization (PSO):** A continuous optimization solver with inertia weight decay and velocity clamping.
    
    - **Ant Colony Optimization (ACO):** A constructive metaheuristic for combinatorial problems (TSP) with pheromone evaporation and local search (2-opt).
    
3. **Neural Networks:**
    
    - **Multi-Layer Perceptron (MLP):** Implemented via **PyTorch** with dynamic hidden layer configuration and Early Stopping.
    
    - **Perceptron:** A single-layer classifier using Hebbian learning rules.
    
    - **Hopfield Network:** An energy-based recurrent network for associative memory and pattern recovery.
    
    - **Self-Organizing Map (SOM):** Unsupervised competitive learning for clustering and dimensionality reduction.
    
4. **Fuzzy Systems:**
    
    - **Fuzzy Controller:** Utilizes **scikit-fuzzy** with Wang-Mendel rule generation for interpretable control/classification.
    

**B. Problem Domains**

Evaluation is conducted on four standardized domains:

1. **Continuous Optimization:** High-dimensional functions (10D, 20D, 30D) including Rastrigin, Ackley, Rosenbrock, Sphere, Schwefel, and Griewank.
    
2. **Combinatorial Optimization:** Traveling Salesman Problem (TSP) using TSPLIB instances (`eil51`, `berlin52`) and random instances solved via Branch-and-Bound (exact) or LKH (heuristic) for baseline truth.
    
3. **Classification:** The Titanic dataset, requiring preprocessing (imputation, encoding) to handle imbalanced classes.
    
4. **Clustering:** Iris (low-dimensional) and Mall Customers (commercial segmentation) datasets for unsupervised evaluation.

---

---

## Section 3:  Setup

### 3.1 Hardware and Software Environment

All experiments were conducted using a standardized software environment to ensure reproducibility.

- **Programming Language:** Python 3.x

- **LLM Backend:** Groq API serving `llama-3.3-70b-versatile`(At the time being ofcourse) with a temperature setting of `0.3` to balance creativity with deterministic output structure.
    
- **Core Libraries:**
    
    - **PyTorch:** For GPU-accelerated neural network training (MLP, Hopfield).
        
    - **DEAP:** For implementing evolutionary structures (GA, GP).
        
    - **Scikit-learn:** For data preprocessing (`StandardScaler`, `LabelEncoder`), splitting, and metric calculation.
        
    - **Scikit-fuzzy:** For fuzzy logic operations.
        
    - **NumPy/Pandas:** For high-performance matrix operations and data manipulation.
        
- **Logging:** A hierarchical logging system records all experiment traces, convergence histories, and agent reasoning to `/outputs/logs`.

### 3.2 Parameter Settings

While MetaMind dynamically suggests parameters based on the specific problem instance, the search space for these parameters is constrained by the `PARAM_SPECS` defined in each method's class. The baseline configuration ranges are as follows:

| **Method** | **Key Parameters**                                                                          | **Range / Options**                                                                    | **Notes**                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **GA**     | `population_size`<br><br>  <br><br>`crossover_rate`<br><br>  <br><br>`mutation_rate`        | [50, 500]<br><br>  <br><br>[0.6, 0.95]<br><br>  <br><br>[0.01, 0.3]                    | Uses Tournament selection and Elitism to preserve best solutions.                               |
| **PSO**    | `n_particles`<br><br>  <br><br>`w` (Inertia)<br><br>  <br><br>`c1`, `c2` (Cognitive/Social) | [20, 200]<br><br>  <br><br>[0.4, 0.9]<br><br>  <br><br>[1.0, 2.5]                      | Includes a linear decay mechanism for inertia weight ($w$) to balance exploration/exploitation. |
| **ACO**    | `n_ants`<br><br>  <br><br>`alpha`, `beta`<br><br>  <br><br>`evaporation_rate`               | [20, 200]<br><br>  <br><br>[0.5, 2.0], [1.0, 5.0]<br><br>  <br><br>[0.1, 0.9]          | Integrated with local search (2-opt) to improve solution quality per iteration.                 |
| **MLP**    | `hidden_layers`<br><br>  <br><br>`learning_rate`<br><br>  <br><br>`optimizer`               | List[int] (dynamic)<br><br>  <br><br>[0.0001, 0.01]<br><br>  <br><br>Adam, SGD         | Dynamic architecture; trained with Early Stopping (patience=50).                                |
| **SOM**    | `map_size`<br><br>  <br><br>`learning_rate`<br><br>  <br><br>`topology`                     | Tuple (e.g., 10x10)<br><br>  <br><br>[0.1, 1.0]<br><br>  <br><br>Hexagonal/Rectangular | Neighborhood radius decays exponentially over epochs.                                           |
| **Fuzzy**  | `n_mfs`<br><br>  <br><br>`defuzzification`                                                  | 3, 5, 7<br><br>  <br><br>Centroid, Bisector                                            | Rules generated automatically via Wang-Mendel method or defined manually.                       |

+ **Optimization Feedback Loop Settings:** The Orchestrator implements an autonomous feedback mechanism defined by the following constraints:

	- **Trigger Condition:** The loop activates only if the `Gap Percentage` (deviation from optimal) exceeds **5.0%**.
	    
	- **Iteration Budget:** A maximum of **2 feedback iterations** is allowed to prevent infinite loops and manage API costs.
	    
	- **History Tracking:** The best solution found across _all_ iterations (initial + feedback) is retained as the final result.


### 3.3 Evaluation Metrics Definitions

Performance is quantified using domain-specific metrics calculated via `src/utils/metrics.py`.

**1. Classification (Titanic):**

- **Accuracy:** The ratio of correctly predicted observations to total observations.
    
- **F1-Score:** The harmonic mean of Precision and Recall, providing a robust metric for the imbalanced survival data.
    
- **AUC-ROC:** Area Under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between classes.
    

**2. Clustering (Iris, Mall Customers):**

- **Silhouette Score:** Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). Range: [-1, 1].

- **Davies-Bouldin Index:** The average similarity measure of each cluster with its most similar cluster. Lower values indicate better clustering.

- **ARI (Adjusted Rand Index):** Used for the Iris dataset to measure agreement between assigned clusters and ground truth labels.


**3. Optimization (TSP, Continuous Functions):**

- **Fitness Value:** The objective function value (minimized tour length for TSP; function output for continuous problems).
    
- **Gap Percentage:** The deviation from the known optimal value, calculated as:
    
    $$Gap(\%) = \frac{|Best_{found} - Optimal|}{|Optimal|} \times 100$$
    
- **Convergence Speed:** Defined as the number of iterations required to reach within 5% of the final best fitness.
    

### 3.4 Statistical Testing Methodology

To ensure that performance differences are statistically significant and not artifacts of stochasticity:

1. **Independent Runs:** Each experiment is repeated `n_runs=5` times with different random seeds.
    
2. **Reporting Standards:** Results are aggregated and reported as "Mean ± Standard Deviation".
    
3. **Confidence Intervals:** 95% confidence intervals are computed for the mean fitness to establish bounds on expected performance.
    
4. **Significance Testing:** The **Wilcoxon Signed-Rank Test** (non-parametric) is employed for pairwise comparisons between methods (e.g., comparing GA vs. PSO on the Rastrigin function). A p-value < 0.05 is required to reject the null hypothesis and claim a significant performance difference.


### 3.6 Prompt Engineering Strategy To ensure high-quality recommendations, a composite prompting strategy was implemented using the `PromptBuilder` module:

- **System Prompt:** Acts as a "CI Architect" persona, embedding the full specification of all 9 methods (`PARAM_SPECS`), including valid ranges and data types. This prevents the LLM from hallucinating non-existent parameters.
    
- **Context Injection:** Problem metadata (dimensions, constraints, known optima) is dynamically injected into the user prompt.
    
- **Feedback Injection:** During the feedback loop, the prompt includes the _previous_ execution metrics and the _gap percentage_, explicitly instructing the model to analyze whether the issue was "exploration" (stuck in local optima) or "exploitation" (slow convergence).