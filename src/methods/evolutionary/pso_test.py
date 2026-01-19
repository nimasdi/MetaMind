import numpy as np
from pso import PSO  

def sphere(x):
    return np.sum(x**2)

n_particles = 30
max_iterations = 300
w = 0.8
c1 = 1.8
c2 = 1.8
w_decay = True
velocity_clamp = 0.4

dimensions_list = [10, 20, 30]

for n_dim in dimensions_list:
    print(f"\n{'='*60}")
    print(f"TESTING SPHERE FUNCTION - DIMENSION: {n_dim}")
    print(f"{'='*60}")

    bounds = [(-5.12, 5.12)] * n_dim

    pso = PSO(
        n_particles=200,
        max_iterations=500,
        w=0.8,
        c1=1.8,
        c2=1.8,
        w_decay=True,
        velocity_clamp=0.8
    )

    results = pso.fit(
        problem_data={'objective_function': sphere, 'bounds': bounds}
    )

    best_position = results['best_position']
    best_score = results['best_fitness']

    print(f"Best solution: {best_position}")
    print(f"Best score (f(x)): {best_score:.10f}")
    print(f"Distance from origin (||x||): {np.linalg.norm(best_position):.6f}")
    print(f"Expected minimum: 0.0")

