import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from methods.neural.som import SOM


def generate_sample_data(n_samples=300, random_seed=909):
    np.random.seed(random_seed)
    
    cluster1 = np.random.randn(n_samples // 3, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(n_samples // 3, 2) * 0.5 + np.array([3, 3])
    cluster3 = np.random.randn(n_samples // 3, 2) * 0.5 + np.array([3, 0])
    
    data = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0] * (n_samples // 3) + 
                     [1] * (n_samples // 3) + 
                     [2] * (n_samples // 3))
    
    indices = np.random.permutation(n_samples)
    data = data[indices]
    labels = labels[indices]
    
    return data, labels


def visualize_som_grid(som):
    grid, map_size = som.get_map_grid()
    height_grid, width_grid, n_features = grid.shape
    
    u_matrix = np.zeros((height_grid, width_grid))
    
    for i in range(height_grid):
        for j in range(width_grid):
            idx = i * width_grid + j
            neighbors = []
            
            if i > 0:
                neighbors.append((i-1) * width_grid + j)
            if i < height_grid - 1:
                neighbors.append((i+1) * width_grid + j)
            if j > 0:
                neighbors.append(i * width_grid + (j-1))
            if j < width_grid - 1:
                neighbors.append(i * width_grid + (j+1))
            
            if neighbors:
                distances = [np.linalg.norm(som.weights[idx] - som.weights[n]) for n in neighbors]
                u_matrix[i, j] = np.mean(distances)
    
    u_min, u_max = u_matrix.min(), u_matrix.max()
    if u_max > u_min:
        u_normalized = (u_matrix - u_min) / (u_max - u_min)
    else:
        u_normalized = u_matrix
    
    print("\n  U-Matrix (Cluster Boundaries):")
    print("  " + "━" * (width_grid * 3 + 2))
    
    for i in range(height_grid):
        row_str = "  ┃"
        for j in range(width_grid):
            val = u_normalized[i, j]
            if val < 0.2:
                char = "  "  # Cluster center
            elif val < 0.4:
                char = "░░"
            elif val < 0.6:
                char = "▒▒"
            elif val < 0.8:
                char = "▓▓"
            else:
                char = "██"  # Boundary
            row_str += char + " "
        print(row_str[:-1] + "┃")
    
    print("  " + "━" * (width_grid * 3 + 2))
    print("  Legend: Empty=Cluster Center, Filled=Boundary")


def visualize_clusters(data, som, labels):
    bmu_indices, _ = som.predict(data)
    height, width = som.results['map_size']
    
    print("\n  Cluster Distribution on Map:")
    
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_bmus = bmu_indices[cluster_mask]
        
        cluster_map = np.zeros((height, width), dtype=int)
        for bmu in cluster_bmus:
            row = bmu // width
            col = bmu % width
            cluster_map[row, col] += 1
        
        print(f"\n  Cluster {cluster_id} ({np.sum(cluster_mask)} samples):")
        print("  " + "━" * (width * 3 + 2))
        
        max_in_cluster = cluster_map.max()
        
        for i in range(height):
            row_str = "  ┃"
            for j in range(width):
                val = cluster_map[i, j]
                if val == 0:
                    row_str += "   "
                else:
                    intensity = val / max_in_cluster if max_in_cluster > 0 else 0
                    if intensity > 0.6:
                        row_str += f"█{cluster_id} "
                    elif intensity > 0.3:
                        row_str += f"▓{cluster_id} "
                    else:
                        row_str += f"░{cluster_id} "
            print(row_str + "┃")
        
        print("  " + "━" * (width * 3 + 2))


def main():
    print("\n[1] Generating Sample Data...")
    data, labels = generate_sample_data(n_samples=300, random_seed=909)
    print(f"    ✓ Created {len(data)} samples with {data.shape[1]} features")
    print(f"    ✓ Data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"    ✓ Number of clusters: {len(np.unique(labels))}")
    
    print("\n[2] Training SOM...")
    som = SOM(
        map_size=(8, 8),
        learning_rate_initial=0.5,
        learning_rate_final=0.01,
        neighborhood_initial=4.0,
        max_epochs=500,
        topology='rectangular'
    )
    
    problem_data = {'X': data}
    results = som.fit(problem_data, random_seed=909)
    
    print(f"\n    Training Results:")
    print(f"    • Map size: {results['map_size'][0]}×{results['map_size'][1]} = {results['map_size'][0] * results['map_size'][1]} neurons")
    print(f"    • Topology: {results['topology']}")
    print(f"    • Training time: {results['training_time']:.4f}s")
    print(f"    • Final quantization error: {results['quantization_error']:.6f}")
    print(f"    • Active neurons: {results['final_active_neurons']}/{results['map_size'][0] * results['map_size'][1]} ({results['final_active_neurons']/(results['map_size'][0] * results['map_size'][1])*100:.1f}%)")
    
    print("\n[3] Sample Predictions...")
    test_samples = data[:10]
    bmu_indices, distances = som.predict(test_samples)
    
    print(f"    First 10 samples mapped to neurons:")
    for i, (bmu_idx, dist) in enumerate(zip(bmu_indices[:5], distances[:5])):
        row = bmu_idx // results['map_size'][1]
        col = bmu_idx % results['map_size'][1]
        print(f"    • Sample {i}: neuron({row},{col}) @ distance {dist:.4f}")
    print(f"    ... (showing 5/{len(test_samples)})")
    
    print("\n[4] SOM Structure Visualization...")
    visualize_som_grid(som)
        
    print("\n[5] Cluster Mapping...")
    visualize_clusters(data, som, labels)
    
    print("\n[6] Analysis...")
    convergence = som.convergence_history
    
    print(f"    • Total epochs: {len(convergence)}")
    print(f"    • Initial error: {convergence[0]:.6f}")
    print(f"    • Final error: {convergence[-1]:.6f}")
    print(f"    • Improvement: {(1 - convergence[-1]/convergence[0])*100:.2f}%")
    
if __name__ == '__main__':
    main()
