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
