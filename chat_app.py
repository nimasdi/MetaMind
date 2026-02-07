import streamlit as st
import os
import sys
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.orchestrator.agent import MetaMindAgent
from src.orchestrator.memory import MemoryManager

# Import all problem types
from src.problems.continuous import (
    RastriginFunction, AckleyFunction, RosenbrockFunction, 
    SphereFunction
)
from src.problems.classification import TitanicProblem
from src.problems.clustering import (
    IrisProblem, MallCustomersProblem, SyntheticClusteringProblem
)
from src.problems.tsp import TSPProblem, load_tsplib_instance, create_random_tsp

# Import all methods
from src.methods.evolutionary.pso import PSO
from src.methods.evolutionary.ga import GeneticAlgorithm
from src.methods.evolutionary.aco import AntColonyOptimization
from src.methods.evolutionary.gp import GeneticProgramming
from src.methods.neural.mlp import MLP
from src.methods.neural.perceptron import Perceptron
from src.methods.neural.hopfield import HopfieldNetwork
from src.methods.neural.som import SOM
from src.methods.fuzzy.controller import FuzzyController

# Utility imports
from src.utils.preprocessing import preprocess_titanic
from src.utils.metrics import compute_gap_percentage

# Page config
st.set_page_config(
    page_title="MetaMind AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat-like appearance
st.markdown("""
<style>
.chat-container {
    max-height: 600px;
    overflow-y: auto;
    padding: 10px;
    border-radius: 10px;
    background-color: #f8f9fa;
    margin-bottom: 20px;
}

.user-message {
    background-color: #007bff;
    color: white;
    padding: 10px 15px;
    border-radius: 15px 15px 5px 15px;
    margin: 10px 0 10px auto;
    max-width: 80%;
    text-align: right;
}

.assistant-message {
    background-color: #e9ecef;
    color: #333;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 5px;
    margin: 10px auto 10px 0;
    max-width: 80%;
}

.system-message {
    background-color: #28a745;
    color: white;
    padding: 8px 12px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: center;
    font-size: 0.9em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager()
    if "current_problem" not in st.session_state:
        st.session_state.current_problem = None
    if "current_problem_type" not in st.session_state:
        st.session_state.current_problem_type = "Continuous Optimization"
    if "solving" not in st.session_state:
        st.session_state.solving = False
    if "comparing" not in st.session_state:
        st.session_state.comparing = False
    if "results_history" not in st.session_state:
        st.session_state.results_history = []

def add_message(role, content, metadata=None):
    """Add a message to chat history."""
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(),
        "metadata": metadata or {}
    }
    st.session_state.chat_history.append(message)

def display_chat_message(message):
    """Display a single chat message."""
    role = message["role"]
    content = message["content"]
    timestamp = message["timestamp"].strftime("%H:%M:%S")
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>You</strong> • {timestamp}<br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "assistant":
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🧠 MetaMind AI</strong> • {timestamp}<br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "system":
        st.markdown(f"""
        <div class="system-message">
            ⚙️ {content}
        </div>
        """, unsafe_allow_html=True)

def create_problem_from_selection(problem_category, problem_type, **kwargs):
    """Create problem instance from user selection."""
    try:
        if problem_category == "Continuous Optimization":
            dimension = kwargs.get('dimension', 10)
            problem_map = {
                "Rastrigin Function": RastriginFunction,
                "Ackley Function": AckleyFunction,
                "Rosenbrock Function": RosenbrockFunction,
                "Sphere Function": SphereFunction,
            }
            
            if problem_type in problem_map:
                problem_class = problem_map[problem_type]
                problem = problem_class(dimension=dimension)
                problem.load_data()
                return problem
        
        elif problem_category == "Classification":
            if problem_type == "Titanic Survival":
                # Load and preprocess Titanic data
                data_dir = project_root / "data" / "titanic_dataset"
                try:
                    clean_data = preprocess_titanic(data_dir, split_ratio=(0.7, 0.15, 0.15))
                    problem = TitanicProblem()
                    problem.X_train = clean_data['X_train']
                    problem.y_train = clean_data['y_train']
                    problem.X_val = clean_data['X_val']
                    problem.y_val = clean_data['y_val']
                    problem.X_test = clean_data['X_test']
                    problem.y_test = clean_data['y_test']
                    problem.feature_names = clean_data['feature_names']
                    return problem
                except Exception as e:
                    st.error(f"Failed to load Titanic dataset: {e}")
                    return None
        
        elif problem_category == "Clustering":
            if problem_type == "Iris Dataset":
                problem = IrisProblem()
                problem.load_data()
                return problem
            elif problem_type == "Mall Customers":
                mall_path = project_root / "data" / "clustering_dataset" / "Mall_Customers.csv"
                if mall_path.exists():
                    problem = MallCustomersProblem()
                    problem.load_data(filepath=str(mall_path))
                    return problem
                else:
                    st.error(f"Mall Customers dataset not found at {mall_path}")
                    return None
            elif problem_type == "Synthetic Clustering":
                n_clusters = kwargs.get('n_clusters', 5)
                n_samples = kwargs.get('n_samples', 500)
                n_features = kwargs.get('n_features', 5)
                problem = SyntheticClusteringProblem(n_clusters=n_clusters)
                problem.load_data(n_samples=n_samples, n_features=n_features, cluster_std=1.0)
                return problem
        
        elif problem_category == "TSP (Traveling Salesman)":
            if problem_type == "Random TSP":
                n_cities = kwargs.get('n_cities', 20)
                problem = create_random_tsp(n_cities=n_cities, seed=42, bounds=(0, 1000))
                return problem
            elif problem_type in ["Berlin52", "EIL51", "KroA100"]:
                tsplib_dir = project_root / "data" / "tsplib"
                instance_map = {
                    "Berlin52": "berlin52",
                    "EIL51": "eil51", 
                    "KroA100": "kroA100"
                }
                instance_name = instance_map[problem_type]
                try:
                    problem = load_tsplib_instance(instance_name, str(tsplib_dir))
                    return problem
                except Exception as e:
                    st.error(f"Failed to load {problem_type}: {e}")
                    return None
        
        return None
    
    except Exception as e:
        st.error(f"Error creating problem: {e}")
        return None

def parse_problem_request(user_input: str) -> tuple:
    """Parse natural language request to identify problem type and parameters."""
    user_input_lower = user_input.lower()
    
    # Continuous Optimization
    if any(keyword in user_input_lower for keyword in ["rastrigin", "optimize rastrigin", "minimize rastrigin"]):
        dimension = extract_number(user_input, default=10)
        return ("Continuous Optimization", "Rastrigin Function", {"dimension": dimension})
    elif any(keyword in user_input_lower for keyword in ["ackley", "optimize ackley"]):
        dimension = extract_number(user_input, default=10)
        return ("Continuous Optimization", "Ackley Function", {"dimension": dimension})
    elif any(keyword in user_input_lower for keyword in ["rosenbrock", "optimize rosenbrock"]):
        dimension = extract_number(user_input, default=10)
        return ("Continuous Optimization", "Rosenbrock Function", {"dimension": dimension})
    elif any(keyword in user_input_lower for keyword in ["sphere function", "optimize sphere"]):
        dimension = extract_number(user_input, default=10)
        return ("Continuous Optimization", "Sphere Function", {"dimension": dimension})
    
    # Classification
    elif any(keyword in user_input_lower for keyword in ["titanic", "classify titanic", "survival prediction"]):
        return ("Classification", "Titanic Survival", {})
    
    # Clustering
    elif any(keyword in user_input_lower for keyword in ["iris", "iris dataset", "iris clustering"]):
        return ("Clustering", "Iris Dataset", {})
    elif any(keyword in user_input_lower for keyword in ["mall customer", "customer segmentation"]):
        return ("Clustering", "Mall Customers", {})
    elif any(keyword in user_input_lower for keyword in ["synthetic cluster", "synthetic clustering"]):
        n_clusters = extract_number(user_input, default=5)
        return ("Clustering", "Synthetic Clustering", {"n_clusters": n_clusters})
    
    # TSP
    elif any(keyword in user_input_lower for keyword in ["tsp", "traveling salesman", "tour"]):
        if any(word in user_input_lower for word in ["random", "generate"]):
            n_cities = extract_number(user_input, default=30)
            return ("TSP (Traveling Salesman)", "Random TSP", {"n_cities": n_cities})
        elif "berlin" in user_input_lower:
            return ("TSP (Traveling Salesman)", "Berlin52", {})
        elif "eil" in user_input_lower:
            return ("TSP (Traveling Salesman)", "EIL51", {})
        elif "kroa" in user_input_lower or "kro" in user_input_lower:
            return ("TSP (Traveling Salesman)", "KroA100", {})
        return ("TSP (Traveling Salesman)", "Random TSP", {"n_cities": 30})
    
    return None, None, {}

def extract_number(text: str, default: int = 10) -> int:
    """Extract first number from text, return default if none found."""
    import re
    numbers = re.findall(r'\b\d+\b', text)
    return int(numbers[0]) if numbers else default

def get_memory_context(agent, problem_type: str, problem_name: str = None) -> str:
    """Get memory context from past optimization runs."""
    memory_manager = st.session_state.memory_manager
    
    # Map problem type to memory category
    category_map = {
        "Continuous Optimization": "continuous_optimization",
        "Classification": "classification",
        "Clustering": "clustering",
        "TSP (Traveling Salesman)": "combinatorial_optimization",
    }
    
    memory_category = category_map.get(problem_type, problem_type)
    
    if problem_name:
        return memory_manager.get_context_string(memory_category, problem_name, top_k=3)
    return ""


def show_tsp_distance_matrix(problem):
    """Save and display TSP distance matrix in chat (scrollable HTML preview)."""
    try:
        dist_mat = getattr(problem, 'distance_matrix', None)
        # Some TSP objects may compute the matrix on demand
        if dist_mat is None and hasattr(problem, 'compute_distance_matrix'):
            try:
                dist_mat = problem.compute_distance_matrix()
            except Exception:
                dist_mat = None

        if dist_mat is None:
            return None

        out_dir = project_root / "outputs" / "memory"
        out_dir.mkdir(parents=True, exist_ok=True)
        instance = getattr(problem, 'instance_name', f"tsp_{getattr(problem, 'n_cities', 'unknown')}")
        csv_path = out_dir / f"distance_matrix_{instance}.csv"
        df_full = pd.DataFrame(dist_mat)
        df_full.to_csv(csv_path, index=False)

        max_preview = 20
        if df_full.shape[0] > max_preview or df_full.shape[1] > max_preview:
            df_preview = df_full.iloc[:max_preview, :max_preview]
            preview_note = f"Showing first {max_preview}x{max_preview} of {df_full.shape[0]}x{df_full.shape[1]} matrix. Full matrix saved to {csv_path}."
        else:
            df_preview = df_full
            preview_note = f"Full matrix saved to {csv_path}."

        try:
            html_table = df_preview.to_html(index=False, float_format='%.3f')
        except Exception:
            html_table = df_preview.head(10).to_html(index=False)

        container_html = (
            f"<strong>Distance matrix (rows = cities):</strong><br>"
            f"<div style='max-height:420px; overflow:auto; border:1px solid #ddd; padding:8px; background:#fff;'>"
            + html_table + "</div>"
        )
        container_html += f"<p style='font-size:0.9em;color:#666;'>{preview_note}</p>"

        add_message("assistant", container_html)
        return str(csv_path)
    except Exception as e:
        st.warning(f"Could not show distance matrix: {e}")
        return None

def process_agent_response(agent, user_input: str, problem):
    """Process user input and get agent response with memory context."""
    
    # Get memory context if we have a problem
    memory_context = ""
    if problem:
        problem_name = getattr(problem, 'problem_name', 'Unknown')
        memory_context = get_memory_context(agent, st.session_state.current_problem_type, problem_name)
    
    # Prepare enhanced prompt with memory
    enhanced_input = user_input
    if memory_context:
        enhanced_input = f"{memory_context}\n\nUser Request: {user_input}"
    
    # Get agent response (this would typically call the agent's chat method)
    # For now, we'll just acknowledge and provide guidance
    return f"I'm analyzing your request about {st.session_state.current_problem_type}. {memory_context if memory_context else 'No previous runs found for reference.'}"

def get_available_methods(problem_category):
    """Get available methods based on problem category."""
    if problem_category == "Continuous Optimization":
        return {
            'PSO': PSO.PARAM_SPECS,
            'GeneticAlgorithm': GeneticAlgorithm.PARAM_SPECS,
        }
    elif problem_category == "Classification":
        return {
            'MLP': MLP.PARAM_SPECS,
            'Perceptron': Perceptron.PARAM_SPECS,
            'FuzzyController': FuzzyController.PARAM_SPECS,
        }
    elif problem_category == "Clustering":
        return {
            'SOM': SOM.PARAM_SPECS,
        }
    elif problem_category == "TSP (Traveling Salesman)":
        return {
            'AntColonyOptimization': AntColonyOptimization.PARAM_SPECS,
            'GeneticAlgorithm': GeneticAlgorithm.PARAM_SPECS,
        }
    else:
        # Default to all methods
        return {
            'PSO': PSO.PARAM_SPECS,
            'GeneticAlgorithm': GeneticAlgorithm.PARAM_SPECS,
            'AntColonyOptimization': AntColonyOptimization.PARAM_SPECS,
            'MLP': MLP.PARAM_SPECS,
            'SOM': SOM.PARAM_SPECS,
            'FuzzyController': FuzzyController.PARAM_SPECS,
        }

def create_method_from_recommendation(recommendation, problem_category):
    """Create method instance from LLM recommendation."""
    method_map = {
        # Optimization methods
        'PSO': PSO,
        'ParticleSwarmOptimization': PSO,
        'Particle Swarm Optimization': PSO,
        'GeneticAlgorithm': GeneticAlgorithm,
        'Genetic Algorithm': GeneticAlgorithm,
        'GA': GeneticAlgorithm,
        
        # TSP methods
        'AntColonyOptimization': AntColonyOptimization,
        'ACO': AntColonyOptimization,
        'Ant Colony Optimization': AntColonyOptimization,
        
        # Neural methods
        'MLP': MLP,
        'MultiLayerPerceptron': MLP,
        'Multi-Layer Perceptron': MLP,
        'Perceptron': Perceptron,
        'HopfieldNetwork': HopfieldNetwork,
        'Hopfield': HopfieldNetwork,
        'SOM': SOM,
        'SelfOrganizingMap': SOM,
        'Self-Organizing Map': SOM,
        
        # Fuzzy methods
        'FuzzyController': FuzzyController,
        'Fuzzy': FuzzyController,
        
        # Genetic Programming
        'GeneticProgramming': GeneticProgramming,
        'GP': GeneticProgramming,
    }
    
    method_class = method_map.get(recommendation.selected_method)
    if not method_class:
        # Fallback based on problem category
        fallback_map = {
            "Continuous Optimization": PSO,
            "Classification": MLP,
            "Clustering": SOM,
            "TSP (Traveling Salesman)": AntColonyOptimization,
        }
        method_class = fallback_map.get(problem_category, PSO)
        st.warning(f"Unknown method '{recommendation.selected_method}', using {method_class.__name__} as fallback")
    
    # Handle parameter type conversions
    params = recommendation.parameters.copy()
    
    # Convert lists to tuples where needed (e.g., map_size for SOM)
    if 'map_size' in params and isinstance(params['map_size'], list):
        params['map_size'] = tuple(params['map_size'])
    if 'hidden_layers' in params and isinstance(params['hidden_layers'], list):
        params['hidden_layers'] = params['hidden_layers']  # Keep as list
    
    return method_class(**params)

def plot_convergence_history(convergence_data, title="Convergence History"):
    """Create a plotly figure for convergence history visualization."""
    fig = go.Figure()
    
    if isinstance(convergence_data, list):
        convergence_data = {"Run 1": convergence_data}
    
    for run_name, history in convergence_data.items():
        if history:
            fig.add_trace(go.Scatter(
                x=list(range(len(history))),
                y=history,
                mode='lines',
                name=run_name,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Fitness Value",
        yaxis_type="log",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig

def solve_problem_interactive(agent, problem):
    with st.container():
        # Create columns for real-time display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
        
        with col2:
            metrics_placeholder = st.empty()
        
        # Step 1: Get recommendation
        status_placeholder.info("Asking AI for method recommendation...")
        add_message("system", "Getting AI recommendation for optimization method...")
        
        try:
            # Get available methods based on problem category
            available_methods = get_available_methods(st.session_state.current_problem_type)
            
            # Get problem context
            if hasattr(problem, 'function_name'):
                context_msg = f"Interactive optimization of {problem.function_name} function"
            elif hasattr(problem, 'problem_name'):
                context_msg = f"Solving {problem.problem_name}"
            else:
                context_msg = f"Solving {st.session_state.current_problem_type} problem"
            
            recommendation = agent.get_recommendation(
                problem_info=problem.get_info(),
                available_methods=available_methods,
                context=context_msg
            )
            
            # Display recommendation in chat
            rec_text = f"""
**Selected Method**: {recommendation.selected_method}  
**Confidence**: {recommendation.confidence:.1%}  
**Expected Performance**: {recommendation.expected_performance.upper()}  

**Reasoning**: {recommendation.reasoning}

**Parameters**:
"""
            for param, value in recommendation.parameters.items():
                rec_text += f"\n• {param}: {value}"
            
            if recommendation.warnings:
                rec_text += "\n\n **Warnings**:"
                for warning in recommendation.warnings:
                    rec_text += f"\n• {warning}"
            
            add_message("assistant", rec_text)
            
            # Step 2: Execute solving
            status_placeholder.info("⚡ Running algorithm...")
            add_message("system", "Executing selected algorithm...")
            
            # Create method instance
            method = create_method_from_recommendation(recommendation, st.session_state.current_problem_type)
            
            # Execute based on problem type
            start_time = time.time()
            if hasattr(problem, 'reset_evaluations'):
                problem.reset_evaluations()
            
            try:
                if st.session_state.current_problem_type == "Continuous Optimization":
                    # Prepare problem data for optimization
                    problem_data = {
                        'objective_function': problem.evaluate,
                        'bounds': [(problem.lower_bounds[i], problem.upper_bounds[i]) 
                                  for i in range(problem.dimension)],
                        'dimension': problem.dimension,
                    }
                    result = method.fit(problem_data)
                    
                elif st.session_state.current_problem_type == "Classification":
                    # For classification problems
                    result = method.fit({'X_train': problem.X_train, 'y_train': problem.y_train})
                    
                    # Evaluate model on validation data if available
                    if hasattr(problem, 'X_val') and hasattr(problem, 'y_val') and hasattr(method, 'predict'):
                        try:
                            y_pred = method.predict(problem.X_val)
                            accuracy = np.mean(y_pred == problem.y_val)
                            result['accuracy'] = accuracy
                        except:
                            pass  # If evaluation fails, continue without accuracy
                    
                elif st.session_state.current_problem_type == "Clustering":
                    # For clustering problems 
                    result = method.fit({'X': problem.X})
                    
                elif st.session_state.current_problem_type == "TSP (Traveling Salesman)":
                    # For TSP problems - pass the problem object directly
                    result = method.fit(problem)
                
                else:
                    st.error(f"Unsupported problem type: {st.session_state.current_problem_type}")
                    return None
            except Exception as e:
                st.error(f"Error during execution: {e}")
                return None
            
            execution_time = time.time() - start_time
            convergence_history = method.convergence_history
            
            # Step 3: Display results
            status_placeholder.success("Optimization completed!")
            
            # Extract metrics from result based on problem type
            metric_value = float('inf')
            metric_name = 'fitness'
            
            if st.session_state.current_problem_type == "TSP (Traveling Salesman)":
                metric_value = result.get('best_length', float('inf'))
                metric_name = 'tour_length'
            elif st.session_state.current_problem_type == "Classification":
                # Try to get accuracy, loss, or other classification metrics
                accuracy = result.get('accuracy', result.get('test_accuracy', result.get('val_accuracy', None)))
                metric_value = accuracy if accuracy is not None else 0
                metric_name = 'accuracy'
            elif st.session_state.current_problem_type == "Clustering":
                # Try to get clustering metrics like inertia or silhouette score
                metric_value = result.get('inertia', result.get('silhouette_score', 0))
                metric_name = 'inertia'
            else:
                # Continuous Optimization
                metric_value = result.get('best_fitness', float('inf'))
                metric_name = 'fitness'
            
            best_fitness = metric_value
            best_solution = result.get('best_solution', result.get('best_tour', result.get('model', None)))
            
            # Calculate metrics
            gap_percent = None
            if problem.optimal_value is not None:
                gap_percent = compute_gap_percentage(best_fitness, problem.optimal_value)
            
            # Display metrics
            with metrics_placeholder.container():
                if st.session_state.current_problem_type == "Continuous Optimization":
                    # Function Optimization Metrics - 2 columns with 3 rows each
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Best Fitness", f"{best_fitness:.6f}")
                        st.metric("Function Evaluations", f"{problem.function_evaluations:,}")
                        st.metric("Execution Time", f"{execution_time:.2f}s")
                    with col_m2:
                        if gap_percent is not None:
                            st.metric("Gap to Optimal", f"{gap_percent:.2f}%")
                        evals_per_sec = problem.function_evaluations / execution_time if execution_time > 0 else 0
                        st.metric("Evals/Second", f"{evals_per_sec:.0f}")
                        initial_fitness = convergence_history[0] if convergence_history else best_fitness
                        improvement = ((initial_fitness - best_fitness) / max(abs(initial_fitness), 1e-10)) * 100
                        st.metric("Improvement (%)", f"{improvement:.2f}%")
                
                elif st.session_state.current_problem_type == "Classification":
                    # Classification Metrics - 2 columns
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Status", "Completed")
                        st.metric("Training Samples", f"{len(problem.X_train):,}")
                    with col_m2:
                        st.metric("Execution Time", f"{execution_time:.2f}s")
                        st.metric("Features", f"{problem.X_train.shape[1]}")
                
                elif st.session_state.current_problem_type == "Clustering":
                    # Clustering Metrics - 2 columns
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Status", "Completed")
                        n_samples = problem.X.shape[0] if hasattr(problem, 'X') and hasattr(problem.X, 'shape') else 0
                        st.metric("Data Points", f"{n_samples:,}")
                    with col_m2:
                        st.metric("Execution Time", f"{execution_time:.2f}s")
                        n_features = problem.X.shape[1] if hasattr(problem, 'X') and hasattr(problem.X, 'shape') else 0
                        st.metric("Features", f"{n_features}")
                
                elif st.session_state.current_problem_type == "TSP (Traveling Salesman)":
                    # TSP Metrics - 2 columns with 3 rows
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Tour Length", f"{best_fitness:.1f}")
                        st.metric("Cities", f"{problem.n_cities}")
                        st.metric("Execution Time", f"{execution_time:.2f}s")
                    with col_m2:
                        if gap_percent is not None:
                            st.metric("Gap to Optimal", f"{gap_percent:.2f}%")
                        st.metric("Iterations", f"{len(convergence_history)}")
                        quality_score = (1 - min(gap_percent, 100) / 100) * 100 if gap_percent is not None else 0
                        st.metric("Solution Quality", f"{quality_score:.1f}%")
            
            # Create result summary for chat
            if st.session_state.current_problem_type == "TSP (Traveling Salesman)":
                result_text = f"""
**TSP Optimization Results**

• **Best Tour Length**: {best_fitness:.1f}
• **Execution Time**: {execution_time:.2f} seconds
• **Iterations**: {len(convergence_history) if convergence_history else 0}
"""
                if gap_percent is not None:
                    result_text += f"• **Gap to Optimal**: {gap_percent:.2f}%\n"
                result_text += f"""
• **Instance**: {getattr(problem, 'instance_name', 'Unknown')}
• **Cities**: {getattr(problem, 'n_cities', 'Unknown')}

The {recommendation.selected_method} algorithm found a tour for the {getattr(problem, 'instance_name', 'TSP')} instance.
"""
            elif st.session_state.current_problem_type == "Classification":
                training_loss = f"{convergence_history[-1]:.4f}" if convergence_history else "N/A"
                result_text = f"""
**Classification Training Complete**

• **Model**: {recommendation.selected_method}
• **Training Loss**: {training_loss}
• **Training Time**: {execution_time:.2f} seconds
• **Dataset**: {problem.problem_name.replace('Classification-', '')}
• **Training Samples**: {len(problem.X_train):,}
• **Features Used**: {problem.X_train.shape[1]}

The model has been trained successfully and is ready for predictions!
"""
            elif st.session_state.current_problem_type == "Clustering":
                n_samples = problem.X.shape[0] if hasattr(problem, 'X') and hasattr(problem.X, 'shape') else 0
                result_text = f"""
**Clustering Complete**

• **Algorithm**: {recommendation.selected_method}
• **Inertia**: {best_fitness:.4f}
• **Execution Time**: {execution_time:.2f} seconds
• **Data Points Clustered**: {n_samples:,}
• **Features**: {problem.X.shape[1] if hasattr(problem, 'X') and hasattr(problem.X, 'shape') else 0}

Clustering results are ready for analysis. Lower inertia indicates more compact clusters.
"""
            else:
                result_text = f"""
**Optimization Results**

• **Best Fitness**: {best_fitness:.6f}
• **Execution Time**: {execution_time:.2f} seconds
"""
                if hasattr(problem, 'function_evaluations'):
                    result_text += f"• **Function Evaluations**: {problem.function_evaluations:,}\n"
                if gap_percent is not None:
                    result_text += f"• **Gap to Optimal**: {gap_percent:.2f}%\n"
                
                result_text += f"""
• **Convergence**: {'Excellent' if len(convergence_history) > 10 else 'Limited data'}

The {recommendation.selected_method} algorithm optimized the {getattr(problem, 'function_name', 'objective')} function in {getattr(problem, 'dimension', 'unknown')}D space.
"""
            
            add_message("assistant", result_text)
            # (Distance matrix will be attached to the result summary below)
            
            # Step 4: Plot convergence
            if convergence_history:
                problem_display_name = getattr(problem, 'function_name', getattr(problem, 'problem_name', 'Problem'))
                fig = plot_convergence_history(
                    {"Current Run": convergence_history},
                    f"{problem_display_name} Convergence"
                )
                progress_placeholder.plotly_chart(fig, use_container_width=True, key="convergence_chart_1")
            
            # Store results with problem-specific metrics
            result_summary = {
                'problem': problem.problem_name,
                'problem_type': st.session_state.current_problem_type,
                'method': recommendation.selected_method,
                'best_fitness': best_fitness,
                'metric_name': metric_name,
                'execution_time': execution_time,
                'gap_percent': gap_percent,
                'timestamp': datetime.now(),
                'convergence_history': convergence_history
            }

            # Attach distance matrix information for TSP now that result_summary exists
            if st.session_state.current_problem_type == "TSP (Traveling Salesman)":
                try:
                    dist_mat = getattr(problem, 'distance_matrix', None)
                    if dist_mat is not None:
                        out_dir = project_root / "outputs" / "memory"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        csv_path = out_dir / f"distance_matrix_{problem.instance_name}.csv"
                        df_full = pd.DataFrame(dist_mat)
                        df_full.to_csv(csv_path, index=False)

                        # Prepare a safe preview for chat: truncate if large
                        max_preview = 20
                        preview_note = ""
                        if df_full.shape[0] > max_preview or df_full.shape[1] > max_preview:
                            df_preview = df_full.iloc[:max_preview, :max_preview]
                            preview_note = f"Showing first {max_preview}x{max_preview} of {df_full.shape[0]}x{df_full.shape[1]} matrix. Full matrix saved to {csv_path}."
                        else:
                            df_preview = df_full

                        try:
                            html_table = df_preview.to_html(index=False, float_format='%.3f')
                        except Exception:
                            html_table = df_preview.head(10).to_html(index=False)

                        container_html = (
                            f"<strong>Distance matrix (rows = cities):</strong><br>"
                            f"<div style='max-height:420px; overflow:auto; border:1px solid #ddd; padding:8px; background:#fff;'>"
                            + html_table + "</div>"
                        )

                        if preview_note:
                            container_html += f"<p style='font-size:0.9em;color:#666;'>{preview_note}</p>"
                        else:
                            container_html += f"<p style='font-size:0.9em;color:#666;'>Full matrix saved to {csv_path}.</p>"

                        add_message("assistant", container_html)
                        result_summary['distance_matrix_csv'] = str(csv_path)
                        # Keep a small preview in the summary for quick reference
                        result_summary['distance_matrix_preview'] = df_preview.values.tolist()
                except Exception as e:
                    st.warning(f"Could not process distance matrix: {e}")
            
            # Add type-specific metrics
            if st.session_state.current_problem_type == "Classification":
                if best_fitness is not None and best_fitness != 0:
                    result_summary['accuracy'] = best_fitness
            elif st.session_state.current_problem_type == "Clustering":
                result_summary['inertia'] = best_fitness
            elif st.session_state.current_problem_type == "TSP (Traveling Salesman)":
                result_summary['tour_length'] = best_fitness
            
            # Step 4: Plot convergence (for optimization problems)
            if convergence_history and st.session_state.current_problem_type == "Continuous Optimization":
                problem_name = getattr(problem, 'function_name', 'Problem')
                fig = plot_convergence_history(
                    {"Current Run": convergence_history},
                    f"{problem_name} Convergence"
                )
                progress_placeholder.plotly_chart(fig, use_container_width=True, key="convergence_chart_2")
            
            st.session_state.results_history.append(result_summary)
            
            # Save to memory for future reference
            try:
                problem_type_map = {
                    "Continuous Optimization": "continuous_optimization",
                    "Classification": "classification",
                    "Clustering": "clustering",
                    "TSP (Traveling Salesman)": "combinatorial_optimization",
                }
                memory_category = problem_type_map.get(st.session_state.current_problem_type, st.session_state.current_problem_type)
                
                memory_entry = {
                    "Problem": problem.problem_name,
                    "Method": recommendation.selected_method,
                    "Parameters": recommendation.parameters,
                    "Fitness": best_fitness,
                    "Timestamp": datetime.now().isoformat()
                }
                
                # Add type-specific metrics for memory
                if st.session_state.current_problem_type == "Classification" and 'accuracy' in result_summary:
                    memory_entry["F1_Score"] = result_summary['accuracy']
                elif st.session_state.current_problem_type == "Clustering" and 'inertia' in result_summary:
                    memory_entry["Silhouette"] = result_summary['inertia']
                elif st.session_state.current_problem_type == "TSP (Traveling Salesman)":
                    memory_entry["Tour_Length"] = result_summary['tour_length']
                    # Save distance matrix CSV for memory and reference
                    try:
                        dist_mat = getattr(problem, 'distance_matrix', None)
                        if dist_mat is not None:
                            out_dir = project_root / "outputs" / "memory"
                            out_dir.mkdir(parents=True, exist_ok=True)
                            csv_path = out_dir / f"distance_matrix_{problem.instance_name}.csv"
                            df_full = pd.DataFrame(dist_mat)
                            df_full.to_csv(csv_path, index=False)
                            memory_entry["Distance_Matrix_CSV"] = str(csv_path)
                            # Also include a small preview for quick inspection in memory
                            max_preview = 20
                            if df_full.shape[0] > max_preview or df_full.shape[1] > max_preview:
                                memory_entry["Distance_Matrix_Preview"] = df_full.iloc[:max_preview, :max_preview].values.tolist()
                            else:
                                memory_entry["Distance_Matrix_Preview"] = df_full.values.tolist()
                    except Exception as e:
                        st.warning(f"Could not save distance matrix to memory: {e}")
                
                st.session_state.memory_manager.save_memory(memory_category, memory_entry)
            except Exception as e:
                st.warning(f"Could not save to memory: {e}")
            
            return result_summary
            
        except Exception as e:
            status_placeholder.error(f"Error during optimization: {str(e)}")
            add_message("assistant", f"Sorry, I encountered an error: {str(e)}")
            return None

def compare_methods_interactive(agent, problem):
    """Compare multiple methods on the same problem."""
    from src.orchestrator.pipeline import Orchestrator
    
    with st.container():
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Step 1: Initialize orchestrator
        status_placeholder.info("🔄 Initializing multi-method comparison...")
        add_message("system", "Setting up comparison of multiple methods...")
        
        try:
            # Get API key
            api_key = os.getenv('GROQ_API_KEY') or st.session_state.get('api_key')
            if not api_key:
                status_placeholder.error("API key not found!")
                add_message("assistant", "❌ I need an API key to compare methods. Please enter it in the sidebar.")
                return None
            
            # Create orchestrator
            orchestrator = Orchestrator(
                api_key=api_key,
                verbose=False,
                enable_feedback_loop=False  # Disable for faster comparison
            )
            
            # Step 2: Determine methods to compare
            status_placeholder.info("🤖 Asking AI to select methods for comparison...")
            add_message("assistant", "Let me select the best methods to compare for this problem...")
            
            # Let the LLM select methods (up to 3)
            comparison_result = orchestrator.compare_methods(
                problem=problem,
                methods=None,  # Let LLM choose
                context=f"Compare the most suitable methods for this {st.session_state.current_problem_type} problem"
            )
            
            # Step 3: Display results
            status_placeholder.success("✅ Comparison complete!")
            
            comparison = comparison_result['comparison']
            
            # Create results message
            methods_tested = [item['method'] for item in comparison['rankings']]
            result_text = f"""
**🏁 Multi-Method Comparison Results**

**Problem**: {problem.problem_name}  
**Methods Tested**: {', '.join(methods_tested)}  
**Total Time**: {comparison_result['total_time']:.2f}s

---

**🏆 Winner: {comparison['best_method']}**  
**Best Fitness**: {comparison['best_fitness']:.6f}

**📊 Rankings:**
"""
            
            for i, item in enumerate(comparison['rankings'], 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                result_text += f"\n{medal} **{item['method']}**"
                result_text += f"\n   • Fitness: {item['fitness']:.6f}"
                result_text += f"\n   • Time: {item['time']:.2f}s"
                if item.get('relative_performance', 0) > 0:
                    result_text += f"\n   • Performance gap: {item['relative_performance']:.2f}%"
                result_text += "\n"
            
            # Add LLM insights if available
            llm_insights = comparison_result.get('llm_insights')
            if llm_insights:
                result_text += f"\n**💡 AI Analysis:**\n\n"
                result_text += f"{llm_insights.get('winner_analysis', 'No analysis available.')}\n\n"
                
                if llm_insights.get('key_insights'):
                    result_text += "**Key Insights:**\n"
                    for insight in llm_insights['key_insights'][:3]:
                        result_text += f"• {insight}\n"
            
            add_message("assistant", result_text)
            
            # Create visualization comparing methods
            comparison_fig = create_comparison_chart(comparison['rankings'])
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Store in history
            st.session_state.results_history.append({
                'problem': problem.problem_name,
                'problem_type': 'Multi-Method Comparison',
                'method': comparison['best_method'],
                'best_fitness': comparison['best_fitness'],
                'execution_time': comparison_result['total_time'],
                'methods_compared': len(comparison['rankings'])
            })
            
            return comparison_result
            
        except Exception as e:
            status_placeholder.error(f"Comparison error: {str(e)}")
            add_message("assistant", f"❌ Sorry, I encountered an error during comparison: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

def create_comparison_chart(rankings):
    """Create a comparison chart for multiple methods."""
    methods = [item['method'] for item in rankings]
    fitnesses = [item['fitness'] for item in rankings]
    times = [item['time'] for item in rankings]
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Fitness Comparison', 'Execution Time Comparison'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add fitness bars
    fig.add_trace(
        go.Bar(
            name='Fitness',
            x=methods,
            y=fitnesses,
            marker_color='lightblue',
            text=[f"{f:.4f}" for f in fitnesses],
            textposition='auto',
        ),
        row=1, col=1
    )
    
    # Add time bars
    fig.add_trace(
        go.Bar(
            name='Time (s)',
            x=methods,
            y=times,
            marker_color='lightcoral',
            text=[f"{t:.2f}s" for t in times],
            textposition='auto',
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Multi-Method Performance Comparison",
        showlegend=False,
        height=400
    )
    
    fig.update_xaxes(title_text="Method", row=1, col=1)
    fig.update_xaxes(title_text="Method", row=1, col=2)
    fig.update_yaxes(title_text="Fitness Value", row=1, col=1)
    fig.update_yaxes(title_text="Execution Time (s)", row=1, col=2)
    
    return fig

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("MetaMind AI Assistant")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Enter your Groq API key"
        )
        
        if api_key and not st.session_state.agent:
            with st.spinner("Initializing AI Agent..."):
                try:
                    st.session_state.agent = MetaMindAgent(
                        api_key=api_key,
                        model="NousResearch/Hermes-3-Llama-3.1-70B",
                        temperature=0.3,
                        verbose=True
                    )
                    st.success("Agent initialized!")
                    add_message("system", "MetaMind AI Agent initialized successfully!")
                except Exception as e:
                    st.error(f"Failed to initialize agent: {e}")
        
        st.divider()
        
        # Problem selection
        st.header("Problem Setup")
        
        # Problem category selection
        problem_category = st.selectbox(
            "Problem Category", 
            ["Continuous Optimization", "Classification", "Clustering", "TSP (Traveling Salesman)"]
        )
        
        st.session_state.current_problem_type = problem_category
        
        # Problem type selection based on category
        if problem_category == "Continuous Optimization":
            problem_type = st.selectbox(
                "Function Type",
                ["Rastrigin Function", "Ackley Function", "Rosenbrock Function", 
                 "Sphere Function"]
            )
            dimension = st.slider("Dimension", 2, 50, 10)
            problem_kwargs = {"dimension": dimension}
            
        elif problem_category == "Classification":
            problem_type = st.selectbox(
                "Dataset",
                ["Titanic Survival"]
            )
            problem_kwargs = {}
            
        elif problem_category == "Clustering":
            problem_type = st.selectbox(
                "Dataset",
                ["Iris Dataset", "Mall Customers", "Synthetic Clustering"]
            )
            if problem_type == "Synthetic Clustering":
                col1, col2 = st.columns(2)
                with col1:
                    n_clusters = st.slider("Clusters", 2, 10, 5)
                    n_samples = st.slider("Samples", 100, 1000, 500)
                with col2:
                    n_features = st.slider("Features", 2, 10, 5)
                problem_kwargs = {"n_clusters": n_clusters, "n_samples": n_samples, "n_features": n_features}
            else:
                problem_kwargs = {}
                
        elif problem_category == "TSP (Traveling Salesman)":
            problem_type = st.selectbox(
                "Instance",
                ["Random TSP", "Berlin52", "EIL51", "KroA100"]
            )
            if problem_type == "Random TSP":
                n_cities = st.slider("Number of Cities", 10, 100, 20)
                problem_kwargs = {"n_cities": n_cities}
            else:
                problem_kwargs = {}
        
        if st.button("Create Problem", type="primary"):
            problem = create_problem_from_selection(problem_category, problem_type, **problem_kwargs)
            if problem:
                st.session_state.current_problem = problem
                st.success(f"Created {getattr(problem, 'problem_name', problem_type)}")
                add_message("user", f"Created {problem_category.lower()} problem: {problem_type}")
                
                # Customized assistant response based on problem category
                if problem_category == "Continuous Optimization":
                    add_message("assistant", f"Great! I'll help you optimize the {problem.function_name} function in {problem_kwargs.get('dimension', 'unknown')}D space. The global optimum is at f(x) = {problem.optimal_value}. Let's find the best method to solve this!")
                elif problem_category == "Classification":
                    add_message("assistant", f"Perfect! I'll help you build a classifier for the {problem_type} dataset. We'll find the best neural network or fuzzy system to predict the target variable with high accuracy.")
                elif problem_category == "Clustering":
                    add_message("assistant", f"Excellent! I'll help you cluster the {problem_type} dataset using self-organizing maps or other clustering methods to discover hidden patterns in the data.")
                elif problem_category == "TSP (Traveling Salesman)":
                    if problem_type == "Random TSP":
                        add_message("assistant", f"Wonderful! I'll solve this {problem_kwargs.get('n_cities', 'unknown')}-city TSP instance using evolutionary algorithms like Ant Colony Optimization to find the shortest route.")
                    else:
                        add_message("assistant", f"Great choice! The {problem_type} is a classic benchmark. I'll use evolutionary algorithms to find the optimal tour through this {getattr(problem, 'n_cities', 'unknown')}-city instance.")
                    # Show distance matrix on creation for TSP problems
                    try:
                        show_tsp_distance_matrix(problem)
                    except Exception:
                        pass
        
        # Current problem info
        if st.session_state.current_problem:
            problem_name = getattr(st.session_state.current_problem, 'problem_name', 'Current Problem')
            st.info(f"**Active Problem**\n\n{problem_name}")
        
        st.divider()
        
    
    # Main chat interface
    col1, col2 = st.columns([1.5, 1.5])
    
    with col1:
        st.header("Chat with MetaMind AI")
        
        # Chat container
        chat_container = st.container(height=500)
        
        with chat_container:
            for message in st.session_state.chat_history:
                display_chat_message(message)
        
        # Chat input
        user_input = st.chat_input(
            "Ask me about optimization problems...",
            disabled=not st.session_state.agent
        )
        
        if user_input:
            add_message("user", user_input)
            
            # Try to parse problem request from natural language
            parsed_category, parsed_type, parsed_kwargs = parse_problem_request(user_input)
            
            # If a problem type is detected in the input, create it automatically
            if parsed_category and parsed_type and ("solve" in user_input.lower() or "optimize" in user_input.lower() or "cluster" in user_input.lower() or "classify" in user_input.lower()):
                with st.spinner(f"Creating {parsed_category}..."):
                    problem = create_problem_from_selection(parsed_category, parsed_type, **parsed_kwargs)
                    if problem:
                        st.session_state.current_problem = problem
                        st.session_state.current_problem_type = parsed_category
                        
                        # Provide context-aware greeting
                        problem_name = getattr(problem, 'problem_name', parsed_type)
                        memory_context = get_memory_context(st.session_state.agent, parsed_category, problem_name)
                        
                        if memory_context:
                            response = f"Perfect! I've created the {problem_name} problem. Based on my memory of past optimization runs:\n{memory_context}\nShould I solve this problem now?"
                        else:
                            response = f"Great! I've set up the {problem_name} problem. Ready to optimize! Would you like me to solve it now?"
                        
                        add_message("assistant", response)
                        # If TSP, show distance matrix immediately after creation
                        try:
                            if parsed_category == "TSP (Traveling Salesman)":
                                show_tsp_distance_matrix(problem)
                        except Exception:
                            pass
                        st.rerun()
            
            # Process user commands
            elif "compare" in user_input.lower() and st.session_state.current_problem:
                st.session_state.comparing = True
                st.rerun()
            elif "solve" in user_input.lower() and st.session_state.current_problem:
                st.session_state.solving = True
                st.rerun()
            elif parsed_category and not ("solve" in user_input.lower() or "optimize" in user_input.lower()):
                # User is asking about a problem type without solving
                add_message("assistant", f"I can help you with {parsed_category}! Would you like me to create a {parsed_type} problem and solve it?")
                st.rerun()
            else:
                # General chat response with memory if available
                if st.session_state.current_problem:
                    problem_name = getattr(st.session_state.current_problem, 'problem_name', 'current problem')
                    memory_context = get_memory_context(st.session_state.agent, st.session_state.current_problem_type, problem_name)
                    if memory_context:
                        response = f"I understand you're working on {problem_name}. {memory_context} What would you like to do next?"
                    else:
                        response = f"I understand! Ready to help with {st.session_state.current_problem_type}. What would you like me to do?"
                    add_message("assistant", response)
                else:
                    add_message("assistant", f"I understand: '{user_input}'. I'm here to help with optimization problems. You can ask me to optimize continuous functions, classify data, cluster datasets, or solve TSP problems. Try saying 'solve rastrigin function' or 'cluster iris dataset'!")
                st.rerun()
    
    with col2:
        st.header("Optimization Status")
        
        # Handle comparison
        if hasattr(st.session_state, 'comparing') and st.session_state.comparing and st.session_state.agent and st.session_state.current_problem:
            with st.spinner("Comparing methods..."):
                result = compare_methods_interactive(
                    st.session_state.agent, 
                    st.session_state.current_problem
                )
                st.session_state.comparing = False
                if result:
                    st.balloons()
        
        # Handle solving
        if st.session_state.solving and st.session_state.agent and st.session_state.current_problem:
            with st.spinner("Optimizing..."):
                result = solve_problem_interactive(
                    st.session_state.agent, 
                    st.session_state.current_problem
                )
                st.session_state.solving = False
                if result:
                    st.balloons()
        
        # Results history
        if st.session_state.results_history:
            st.subheader("Recent Results")
            
            results_df = pd.DataFrame([
                {
                    'Problem': r.get('problem', 'Unknown'),
                    'Type': r.get('problem_type', 'Unknown'),
                    'Method': r.get('method', 'Unknown'),
                    'Score': (
                        f"{(r.get('accuracy') or 0):.1%}" if r.get('problem_type') == 'Classification' and r.get('accuracy') is not None and r.get('accuracy') > 0
                        else "Model Trained" if r.get('problem_type') == 'Classification'
                        else f"{(r.get('tour_length') or 0):.1f}" if r.get('problem_type') == 'TSP (Traveling Salesman)' and r.get('tour_length') is not None
                        else f"{(r.get('inertia') or 0):.2f}" if r.get('problem_type') == 'Clustering' and r.get('inertia') is not None
                        else f"{(r.get('best_fitness') or 0):.4f}" if r.get('best_fitness') is not None and r.get('best_fitness') != float('inf')
                        else "N/A"
                    ),
                    'Metric': (
                        "Accuracy" if r.get('problem_type') == 'Classification'
                        else "Tour Length" if r.get('problem_type') == 'TSP (Traveling Salesman)'
                        else "Inertia" if r.get('problem_type') == 'Clustering'
                        else "Fitness"
                    ),
                    'Time': f"{r.get('execution_time', 0):.1f}s"
                }
                for r in st.session_state.results_history[-5:]  # Last 5 results
            ])
            
            st.dataframe(results_df, use_container_width=True)
        
        # Performance metrics
        if len(st.session_state.results_history) > 1:
            st.subheader("Performance Trends")
            
            fitness_history = [r['best_fitness'] for r in st.session_state.results_history]
            time_history = [r['execution_time'] for r in st.session_state.results_history]
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                y=fitness_history,
                mode='lines+markers',
                name='Best Fitness',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            fig_perf.update_layout(
                title="Fitness Over Experiments",
                yaxis_title="Fitness Value",
                yaxis_type="log",
                template="plotly_white",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_perf, use_container_width=True, key="performance_trends_chart")

    # Footer
    st.divider()
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        if st.session_state.agent:
            stats = st.session_state.agent.get_stats()
            st.metric("API Calls", stats['total_calls'])
    
    with col_f2:
        st.metric("Problems Solved", len(st.session_state.results_history))
    
    with col_f3:
        if st.session_state.results_history:
            avg_time = np.mean([r['execution_time'] for r in st.session_state.results_history])
            st.metric("Avg. Time", f"{avg_time:.1f}s")

if __name__ == "__main__":
    main()