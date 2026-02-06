import numpy as np
from scipy import stats
from typing import List, Dict, Any


def compute_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    
    values = np.array(values)
    
    # Handle NaN/inf values in statistics
    with np.errstate(invalid='ignore'):
        std_value = np.std(values)
        var_value = np.var(values)
    
    if np.isnan(std_value):
        std_value = 0.0
    if np.isnan(var_value):
        var_value = 0.0
    
    return {
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(std_value),
        'var': float(var_value),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'count': len(values)
    }


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
    if not values or len(values) < 2:
        return (None, None)
    
    values = np.array(values)
    mean = np.mean(values)
    sem = stats.sem(values)
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
    
    return (mean - interval, mean + interval)


def wilcoxon_test(sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
    if len(sample1) != len(sample2):
        raise ValueError("Samples must have the same length")
    
    if len(sample1) < 2:
        return {'statistic': None, 'p_value': None, 'significant': None}
    
    try:
        statistic, p_value = stats.wilcoxon(sample1, sample2)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'alpha': 0.05
        }
    except Exception as e:
        return {'statistic': None, 'p_value': None, 'significant': None, 'error': str(e)}


def compute_gap_percentage(value: float, optimal: float) -> float:
    if abs(optimal) < 1e-10:
        # If both are essentially zero, gap is 0
        if abs(value) < 1e-10:
            return 0.0
        # For zero-optimal problems, express absolute error as percentage of 1.0
        # This gives more reasonable percentages for benchmark functions
        return abs(value) * 100
    
    # Standard relative percentage error for non-zero optimal values
    return abs((value - optimal) / optimal) * 100


def success_rate(values: List[float], threshold: float) -> float:
    if not values:
        return 0.0
    
    successes = sum(1 for v in values if v <= threshold)
    return (successes / len(values)) * 100


def pairwise_wilcoxon_comparison(data_dict: Dict[str, List[float]]) -> Dict[str, Any]:
    methods = list(data_dict.keys())
    n_methods = len(methods)
    
    # Initialize matrices for p-values and significance
    p_values = np.zeros((n_methods, n_methods))
    significant = np.zeros((n_methods, n_methods), dtype=bool)
    effect_sizes = np.zeros((n_methods, n_methods))
    
    # Bonferroni correction
    alpha = 0.05
    n_comparisons = n_methods * (n_methods - 1) / 2
    alpha_corrected = alpha / n_comparisons
    
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                p_values[i, j] = 1.0
                effect_sizes[i, j] = 0.0
                significant[i, j] = False
            elif i < j:
                sample1 = np.array(data_dict[methods[i]])
                sample2 = np.array(data_dict[methods[j]])
                
                # Handle different lengths by truncating to smaller length
                min_len = min(len(sample1), len(sample2))
                if min_len > 0:
                    sample1 = sample1[:min_len]
                    sample2 = sample2[:min_len]
                    
                    try:
                        statistic, p_value = stats.wilcoxon(sample1, sample2)
                        p_values[i, j] = float(p_value)
                        p_values[j, i] = float(p_value)
                        
                        # Bonferroni-corrected significance
                        significant[i, j] = p_value < alpha_corrected
                        significant[j, i] = p_value < alpha_corrected
                        
                        # Compute effect size (rank-biserial correlation)
                        n = len(sample1)
                        r = 1 - (2 * statistic) / (n * (n + 1))
                        effect_sizes[i, j] = abs(r)
                        effect_sizes[j, i] = abs(r)
                    except Exception:
                        p_values[i, j] = np.nan
                        p_values[j, i] = np.nan
                        effect_sizes[i, j] = np.nan
                        effect_sizes[j, i] = np.nan
    
    return {
        'p_values': p_values,
        'significant': significant,
        'effect_sizes': effect_sizes,
        'methods': methods,
        'alpha': alpha,
        'alpha_corrected': alpha_corrected,
        'n_comparisons': n_comparisons
    }


def format_wilcoxon_table(comparison_results: Dict[str, Any]) -> str:
    methods = comparison_results['methods']
    p_values = comparison_results['p_values']
    significant = comparison_results['significant']
    
    header = "Method 1 vs Method 2".ljust(30) + "P-Value".ljust(12) + "Significant*"
    lines = [header, "-" * 60]
    
    # Body - only unique pairs
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            p_val = p_values[i, j]
            sig = significant[i, j]
            
            comparison_str = f"{methods[i]} vs {methods[j]}".ljust(30)
            if np.isnan(p_val):
                p_val_str = "N/A".ljust(12)
            else:
                p_val_str = f"{p_val:.4f}".ljust(12)
            
            sig_str = "***" if sig else "ns"
            lines.append(comparison_str + p_val_str + sig_str)
    
    lines.append("-" * 60)
    lines.append("* Asterisks (***) indicate significant difference (Bonferroni-corrected)")
    lines.append(f"  Significance level: alpha={comparison_results['alpha_corrected']:.6f} (corrected)")
    
    return "\n".join(lines)


def print_wilcoxon_summary(comparison_results: Dict[str, Any], title: str = "Statistical Comparison"):
    print("\n" + "="*70)
    print(f"{title.center(70)}")
    print("="*70)
    print(f"Number of comparisons: {int(comparison_results['n_comparisons'])}")
    print(f"Original alpha: {comparison_results['alpha']}")
    print(f"Bonferroni-corrected alpha: {comparison_results['alpha_corrected']:.6f}")
    print()
    print(format_wilcoxon_table(comparison_results))
    print("="*70 + "\n")
