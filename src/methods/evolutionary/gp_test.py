import numpy as np
import matplotlib.pyplot as plt

from gp import GeneticProgramming


def test_complex_polynomial():
    print("Test 1: Polynomial (f(x) = x^3 - 2x^2 + 3x - 1)")
    
    np.random.seed(42)
    X_train = np.linspace(-3, 3, 100)
    y_train = X_train**3 - 2*X_train**2 + 3*X_train - 1 + np.random.normal(0, 0.3, 100)
    
    gp = GeneticProgramming(
        population_size=500,
        generations=150,
        max_depth=10,
        crossover_rate=0.9,
        mutation_rate=0.15,
        function_set=['+', '-', '*', '/', 'pow2', 'pow3'],
        parsimony_coefficient=0.0001
    )
    
    problem_data = {'X': X_train, 'y': y_train}
    gp.fit(problem_data)
    
    X_test = np.linspace(-4, 4, 150)
    y_pred = gp.predict(X_test)
    y_true = X_test**3 - 2*X_test**2 + 3*X_test - 1
    
    print(f"\nBest Expression: {gp.get_expression()}")
    print(f"Final MSE: {gp.results['final_mse']:.4f}")
    print(f"Tree Depth: {gp.results['tree_depth']}")
    print(f"Tree Size: {gp.results['tree_size']}")
    print(f"Training Time: {gp.results['training_time']:.2f}s")
    
    test_mse = np.mean((y_pred - y_true)**2)
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Convergence length: {len(gp.convergence_history)}")
    print(f"First 5 fitness values: {gp.convergence_history[:5]}")
    print(f"Last 5 fitness values: {gp.convergence_history[-5:]}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.6, label='Training Data', s=30)
    plt.plot(X_test, y_true, 'g--', label='True Function', linewidth=2)
    plt.plot(X_test, y_pred, 'r-', label='GP Prediction', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Regression (x^3 - 2x^2 + 3x - 1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(gp.convergence_history, linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (MSE)')
    plt.title('Convergence History')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    return gp


def test_complex_trigonometric():
    print("Test 2: Trigonometric (f(x) = 2*sin(x) + cos(x) - 0.5)")
    
    np.random.seed(123)
    X_train = np.linspace(-2*np.pi, 2*np.pi, 120)
    y_train = 2*np.sin(X_train) + np.cos(X_train) - 0.5 + np.random.normal(0, 0.15, 120)
    
    gp = GeneticProgramming(
        population_size=500,
        generations=150,
        max_depth=10,
        crossover_rate=0.9,
        mutation_rate=0.15,
        function_set=['+', '-', '*', '/', 'sin', 'cos'],
        parsimony_coefficient=0.0001
    )
    
    problem_data = {'X': X_train, 'y': y_train}
    gp.fit(problem_data)
    
    X_test = np.linspace(-3*np.pi, 3*np.pi, 200)
    y_pred = gp.predict(X_test)
    y_true = 2*np.sin(X_test) + np.cos(X_test) - 0.5
    
    print(f"\nBest Expression: {gp.get_expression()}")
    print(f"Final MSE: {gp.results['final_mse']:.4f}")
    print(f"Tree Depth: {gp.results['tree_depth']}")
    print(f"Tree Size: {gp.results['tree_size']}")
    print(f"Training Time: {gp.results['training_time']:.2f}s")
    
    test_mse = np.mean((y_pred - y_true)**2)
    print(f"Test MSE: {test_mse:.4f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.6, label='Training Data', s=20)
    plt.plot(X_test, y_true, 'g--', label='True Function', linewidth=2)
    plt.plot(X_test, y_pred, 'r-', label='GP Prediction', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trigonometric Regression (2sin(x) + cos(x) - 0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(gp.convergence_history, linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (MSE)')
    plt.title('Convergence History')
    plt.yscale('log')  # Use log scale for better visibility
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    return gp


def main():    
    test_complex_polynomial()
    test_complex_trigonometric()


if __name__ == "__main__":
    main()
