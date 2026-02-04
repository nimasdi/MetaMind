#!/usr/bin/env python3
"""Quick test script for ACO on TSP instances."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only what we need to avoid dependency issues
import importlib.util

# Load ACO directly
spec = importlib.util.spec_from_file_location("aco", "src/methods/evolutionary/aco.py")
aco_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aco_module)
AntColonyOptimization = aco_module.AntColonyOptimization

# Load TSP directly
spec2 = importlib.util.spec_from_file_location("tsp", "src/problems/tsp.py")
tsp_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(tsp_module)
load_tsplib_instance = tsp_module.load_tsplib_instance

def test_aco():
    # Load a TSP instance
    print("Loading berlin52 TSP instance...")
    problem = load_tsplib_instance('berlin52', data_dir='data/tsplib')
    print(f"Problem: {problem}")
    print(f"Optimal known value: {problem.optimal_value}")
    
    # Create ACO instance with default parameters
    print("\nRunning ACO with default parameters...")
    aco = AntColonyOptimization()
    results = aco.fit(problem)
    
    # Display results
    print(f"\n{'='*60}")
    print("ACO Results:")
    print(f"{'='*60}")
    best_len = results['best_length']
    print(f"Best tour length: {best_len:.2f}")
    print(f"Best known optimal: {problem.optimal_value}")
    
    gap = ((best_len - problem.optimal_value) / problem.optimal_value) * 100
    print(f"Gap to optimal: {gap:.2f}%")
    print(f"Iterations: {results['n_iterations']}")
    print(f"Ants: {results['n_ants']}")
    print(f"Execution time: {results['execution_time']:.3f}s")
    
    # Test with custom parameters
    print(f"\n{'='*60}")
    print("Running ACO with custom parameters (more iterations, 2-opt enabled)...")
    aco2 = AntColonyOptimization(
        n_ants=30,
        max_iterations=300,
        alpha=1.2,
        beta=3.0,
        evaporation_rate=0.3,
        local_search=True
    )
    results2 = aco2.fit(problem)
    best_len2 = results2['best_length']
    gap2 = ((best_len2 - problem.optimal_value) / problem.optimal_value) * 100
    print(f"Best tour length: {best_len2:.2f}")
    print(f"Gap to optimal: {gap2:.2f}%")
    print(f"Execution time: {results2['execution_time']:.3f}s")
    
    print("\nTest completed successfully!")

if __name__ == '__main__':
    test_aco()
