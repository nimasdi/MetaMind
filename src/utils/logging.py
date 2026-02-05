import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_name, output_dir="outputs/logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"{experiment_name}_{timestamp}.log"
    
    return setup_logger(experiment_name, str(log_file))


def standard_progress_callback(metrics):
    method = metrics.get('method', 'Method')

    BAR_LEN = 30
    FILL_CHAR = '='
    EMPTY_CHAR = '-'
    ARROW = '>'  

    current = 0
    total = 0
    unit = "Step"
    info_text = ""
    
    # Logic for Epoch-based methods (MLP, Perceptron, SOM, etc.)
    if 'epoch' in metrics:
        current = metrics['epoch']
        total = metrics['max_epochs']
        unit = "Epoch"
        
        if 'train_loss' in metrics:
            info_text += f"Loss: {metrics['train_loss']:.4f} "
        if 'val_accuracy' in metrics:
            info_text += f"Val Acc: {metrics['val_accuracy']:.4f} "
        if 'quantization_error' in metrics:
            info_text += f"QE: {metrics['quantization_error']:.4f} "
            
    # Logic for Generation-based methods (GA, GP, etc.)
    elif 'generation' in metrics:
        current = metrics['generation']
        total = metrics['max_generations']
        unit = "Gen"
        
        if 'best_fitness' in metrics:
            info_text += f"Best Fit: {metrics['best_fitness']:.4f} "

    # Logic for Iteration-based methods (PSO, ACO, Hopfield)
    elif 'iteration' in metrics:
        current = metrics['iteration']
        total = metrics['max_iterations']
        unit = "Iter"
        
        if 'global_best_fitness' in metrics:
            info_text += f"Best: {metrics['global_best_fitness']:.4f} "
        if 'best_tour_length' in metrics:
            info_text += f"Length: {metrics['best_tour_length']:.2f} "
        if 'energy' in metrics:
            info_text += f"Energy: {metrics['energy']:.4f} "

    elif 'status' in metrics and metrics['status'] == 'complete':
        sys.stdout.write(f"\r[DONE] [{method}] Execution Complete. Rules: {metrics.get('n_rules_generated', 'N/A')}          \n")
        sys.stdout.flush()
        return
    
    if total > 0:
        percent = current / total
        filled_len = int(BAR_LEN * percent)
        
        if filled_len < BAR_LEN:
            bar = FILL_CHAR * filled_len + ARROW + EMPTY_CHAR * (BAR_LEN - filled_len - 1)
        else:
            bar = FILL_CHAR * BAR_LEN
            
        if current == 1 or current == total or current % max(1, total // 100) == 0:
            
            msg = (
                f"\r[{method[:12]:<12}] "      
                f"[{bar}] {percent:>4.0%} "    
                f"| {unit} {current}/{total} " 
                f"| {info_text}"               
            )
            
            sys.stdout.write(msg)
            sys.stdout.flush()
