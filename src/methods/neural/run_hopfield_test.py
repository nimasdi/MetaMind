import numpy as np
from hopfield import HopfieldNetwork

def create_test_patterns():
    
    pattern1 = np.array([
        1, -1, -1, -1,
        1, -1, -1, -1,
        1, -1, -1, -1,
        1, -1, -1, -1
    ])
    
    pattern2 = np.array([
        1, 1, 1, 1,
        -1, -1, -1, -1,
        -1, -1, -1, -1,
        -1, -1, -1, -1
    ])
    
    pattern3 = np.array([
        1, -1, -1, -1,
        -1, 1, -1, -1,
        -1, -1, 1, -1,
        -1, -1, -1, 1
    ])
    
    patterns = np.vstack([pattern1, pattern2, pattern3])
    return patterns


def visualize_pattern(pattern, width=4):
    pattern = pattern.reshape(width, -1)
    chars = {1: '█', -1: '·'}
    for row in pattern:
        print('  ' + ' '.join(chars[int(val)] for val in row))


def add_noise(pattern, noise_level=0.2):
    noisy = pattern.copy()
    n_flips = int(len(pattern) * noise_level)
    if n_flips > 0:
        indices = np.random.choice(len(pattern), size=n_flips, replace=False)
        noisy[indices] *= -1
    return noisy


def main():
    np.random.seed(12)
    
    patterns = create_test_patterns()
    
    for i, pattern in enumerate(patterns):
        print(f"\n   Pattern {i+1}:")
        visualize_pattern(pattern)
    
    hopfield = HopfieldNetwork(
        max_iterations=100,
        threshold=0.0,
        async_update=True,
        energy_threshold=1e-6
    )
    
    hopfield.fit({'patterns': patterns})
    
    results = hopfield.get_results()
    print(f"\n   Training Results:")
    print(f"   - Number of patterns: {results['n_patterns']}")
    print(f"   - Network size: {results['n_neurons']} neurons")
    print(f"   - Storage capacity: ~{results['storage_capacity']:.1f} patterns")
    
    for noise_level in [0.1, 0.2, 0.3]:        
        for i, original_pattern in enumerate(patterns):
            noisy_pattern = add_noise(original_pattern, noise_level)
            
            retrieved, energy_history = hopfield.predict(
                noisy_pattern.reshape(1, -1), 
                return_energy=True
            )
            retrieved = retrieved[0]
            energy_history = energy_history[0] 
            
            success = np.allclose(retrieved, original_pattern)
            
            print(f"\n   Pattern {i+1} - {'SUCCESS' if success else 'FAILED'}")
            print(f"   Convergence: {len(energy_history)} iterations")
            print(f"   Final energy: {energy_history[-1]:.6f}")
            
            if not success:
                print("\n   Original:")
                visualize_pattern(original_pattern)
                print("\n   Noisy input:")
                visualize_pattern(noisy_pattern)
                print("\n   Retrieved:")
                visualize_pattern(retrieved)
    

    test_pattern = patterns[0]
    print("\n   Original Pattern 1:")
    visualize_pattern(test_pattern)
    
    noisy_test = add_noise(test_pattern, 0.25)
    print("\n   Noisy version (25% noise):")
    visualize_pattern(noisy_test)
    
    retrieved, energy_history = hopfield.predict(
        noisy_test.reshape(1, -1),
        return_energy=True
    )
    retrieved = retrieved[0]
    energy_history = energy_history[0] 
    
    print("\n   Retrieved pattern:")
    visualize_pattern(retrieved)
    
    print(f"\n   Energy trajectory: {energy_history[0]:.4f} → {energy_history[-1]:.4f}")
    print(f"   Iterations: {len(energy_history)}")
    print(f"   Match: {'YES' if np.allclose(retrieved, test_pattern) else 'NO'}")
    
    print("\nExecution Logs:")
    logs = hopfield.get_logs()
    for log in logs[-5:]:
        print(f"   [{log['elapsed']:.4f}s] {log['message']}")
    
if __name__ == '__main__':
    main()
