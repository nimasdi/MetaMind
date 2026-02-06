# MetaMind User Guide

## Installation

Before using MetaMind, install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install all necessary packages needed to run the chat app and experiments.

## Getting Started

MetaMind provides two main ways to interact with the system:

### 1. Chat App

Run the interactive chat application:

```bash
python chat_app.py
```

This launches an interactive chat interface where you can ask questions and interact with the MetaMind orchestrator in real-time.

### 2. Run Experiments

Explore the `experiments/` directory to run various benchmark tests and examples:

```bash
cd experiments
python run_classification.py      # Run classification experiments
python run_clustering_benchmark.py # Run clustering benchmarks
python run_function_optimization.py # Run function optimization tests
python run_tsp_benchmark.py       # Run traveling salesman problem benchmarks
python multi_method_example.py    # Run multi-method examples
```

Each experiment script demonstrates different capabilities and benchmarks of the system.

## Project Structure

- **chat_app.py** - Interactive chat interface
- **main.py** - Main entry point
- **experiments/** - Collection of benchmark and example scripts
- **src/** - Core library code including methods, problems, and orchestrator
- **data/** - Datasets for experiments (Iris, Mall Customers, TSP instances, Titanic)
- **outputs/** - Results, figures, logs, and memory files from experiments

## For More Information

See [documentation](doc/report.md) for detailed project documentation.
