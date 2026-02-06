import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_convergence(convergence_history: List[float], 
                     title: str = "Convergence History",
                     xlabel: str = "Iteration",
                     ylabel: str = "Fitness",
                     save_path: Optional[str] = None,
                     show: bool = True):
    """
    Plot convergence history of an optimization algorithm.
    
    Args:
        convergence_history: List of fitness values over iterations
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history, linewidth=2, color='#2E86AB')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_convergence(convergence_data: Dict[str, List[float]],
                              title: str = "Convergence Comparison",
                              xlabel: str = "Iteration",
                              ylabel: str = "Fitness",
                              save_path: Optional[str] = None,
                              show: bool = True):
    """
    Plot multiple convergence histories for comparison.
    
    Args:
        convergence_data: Dictionary mapping method names to convergence histories
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for i, (method_name, history) in enumerate(convergence_data.items()):
        color = colors[i % len(colors)]
        plt.plot(history, linewidth=2, label=method_name, color=color, alpha=0.8)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_with_bands(convergence_histories: Dict[str, List[List[float]]],
                                title: str = "Convergence with Confidence Bands",
                                xlabel: str = "Iteration",
                                ylabel: str = "Fitness",
                                confidence: float = 0.95,
                                save_path: Optional[str] = None,
                                show: bool = True):

    plt.figure(figsize=(12, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for i, (name, histories) in enumerate(convergence_histories.items()):
        # Compute mean and std at each iteration
        max_iters = max(len(h) for h in histories)
        means = []
        stds = []
        
        for iter_idx in range(max_iters):
            values = [h[iter_idx] for h in histories if iter_idx < len(h)]
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(np.nan)
                stds.append(0)
        
        means = np.array(means)
        stds = np.array(stds)
        iterations = np.arange(len(means))
        
        color = colors[i % len(colors)]
        
        # Compute confidence interval bounds
        if confidence > 0:
            # Use confidence interval (approximately 1.96 * std for 95%)
            z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.0
            bounds = z_score * stds
            label_text = f"{name} (95% CI)"
        else:
            # Use standard deviation
            bounds = stds
            label_text = f"{name} (±std)"
        
        # Plot mean line
        plt.plot(iterations, means, linewidth=2.5, color=color, label=label_text, zorder=3)
        
        # Plot confidence band
        plt.fill_between(iterations, 
                         means - bounds, 
                         means + bounds,
                         alpha=0.2, 
                         color=color, 
                         zorder=1)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_table(results: List[Dict[str, Any]],
                          save_path: Optional[str] = None):
    """
    Create a visual comparison table of results.
    
    Args:
        results: List of result dictionaries
        save_path: Optional path to save the figure
    """
    if not results:
        return
    
    # Extract data for table
    problems = [r['problem'] for r in results]
    methods = [r['method'] for r in results]
    best_fitness = [r['best_fitness']['min'] for r in results]
    mean_fitness = [r['best_fitness']['mean'] for r in results]
    std_fitness = [r['best_fitness']['std'] for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(results) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for i in range(len(results)):
        gap = ""
        if results[i].get('gap_percent') and results[i]['gap_percent']['mean'] is not None:
            gap = f"{results[i]['gap_percent']['mean']:.2f}%"
        
        row = [
            problems[i],
            methods[i],
            f"{best_fitness[i]:.2f}",
            f"{mean_fitness[i]:.2f} ± {std_fitness[i]:.2f}",
            gap,
            f"{results[i]['computation_time']['mean']:.2f}s"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Problem', 'Method', 'Best', 'Mean ± Std', 'Gap %', 'Time'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.12, 0.2, 0.12, 0.12]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('TSP Benchmark Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Table saved to: {save_path}")
    
    plt.show()


def plot_tsp_tour(cities: np.ndarray, tour: List[int],
                  title: str = "TSP Tour",
                  save_path: Optional[str] = None,
                  show: bool = True):
    """
    Plot a TSP tour.
    
    Args:
        cities: Array of city coordinates (n_cities x 2)
        tour: List of city indices representing the tour
        title: Plot title
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 10))
    
    # Plot cities
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=3, alpha=0.8)
    
    # Plot tour
    tour_cities = cities[tour + [tour[0]]]
    plt.plot(tour_cities[:, 0], tour_cities[:, 1], 'b-', linewidth=1.5, alpha=0.7)
    
    # Add city labels
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='bottom')
    
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tour plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_box_comparison(data_dict: Dict[str, List[float]],
                       title: str = "Performance Comparison",
                       ylabel: str = "Fitness",
                       save_path: Optional[str] = None,
                       show: bool = True):
    """
    Create box plots for comparing multiple methods.
    
    Args:
        data_dict: Dictionary mapping method names to lists of values
        title: Plot title
        ylabel: Y-axis label
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 6))
    
    data = list(data_dict.values())
    labels = list(data_dict.keys())
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    
    # Color boxes
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Box plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
