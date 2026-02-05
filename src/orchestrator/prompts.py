"""
Prompt templates for MetaMind orchestrator.
Contains system prompts and context builders for LLM-based method selection.
"""

import json
from typing import Dict, Any, List
from src.core.types import ProblemType, MethodCategory, OptimizationType


class PromptBuilder:
    """Builds comprehensive prompts for the MetaMind agent."""
    
    METHOD_DESCRIPTIONS = {
        "ACO": {
            "name": "Ant Colony Optimization",
            "category": MethodCategory.SWARM_INTELLIGENCE.value,
            "description": "Bio-inspired metaheuristic for combinatorial optimization. Simulates pheromone trails.",
            "best_for": ["TSP", "routing", "scheduling", "combinatorial problems"],
            "strengths": ["Excellent for graph-based problems", "Fast convergence", "Distributed nature"],
            "weaknesses": ["Can get stuck in local optima", "Requires proper parameter tuning"],
        },
        "GA": {
            "name": "Genetic Algorithm",
            "category": MethodCategory.EVOLUTIONARY.value,
            "description": "Population-based evolutionary algorithm using selection, crossover, and mutation.",
            "best_for": ["optimization", "function fitting", "symbolic regression", "feature selection"],
            "strengths": ["General-purpose", "Handles discrete and continuous", "Parallelizable"],
            "weaknesses": ["Slow on high-dimensional problems", "Needs proper population sizing"],
        },
        "GP": {
            "name": "Genetic Programming",
            "category": MethodCategory.EVOLUTIONARY.value,
            "description": "Evolves tree-based mathematical expressions for symbolic regression.",
            "best_for": ["symbolic regression", "function approximation", "expression discovery"],
            "strengths": ["Interpretable results", "No need to specify form", "Automatic feature discovery"],
            "weaknesses": ["Slow", "Code bloat", "Requires large populations"],
        },
        "PSO": {
            "name": "Particle Swarm Optimization",
            "category": MethodCategory.SWARM_INTELLIGENCE.value,
            "description": "Simulates social behavior of bird flocking or fish schooling.",
            "best_for": ["continuous optimization", "function optimization", "multi-modal problems"],
            "strengths": ["Fast convergence", "Few parameters", "Good for continuous spaces"],
            "weaknesses": ["Can diverge", "Limited for discrete problems"],
        },
        "FuzzyController": {
            "name": "Fuzzy Logic Controller",
            "category": MethodCategory.FUZZY_SYSTEM.value,
            "description": "Uses fuzzy sets and rules for classification and control.",
            "best_for": ["classification", "control systems", "time series", "decision support"],
            "strengths": ["Interpretable rules", "Handles imprecision", "Robust"],
            "weaknesses": ["Requires domain knowledge", "Manual rule definition", "Not best for pure prediction"],
        },
        "Hopfield": {
            "name": "Hopfield Network",
            "category": MethodCategory.NEURAL_NETWORK.value,
            "description": "Recurrent neural network for pattern completion and associative memory.",
            "best_for": ["pattern recognition", "pattern completion", "memory retrieval", "optimization"],
            "strengths": ["Associative memory", "Pattern completion", "Energy-based stability"],
            "weaknesses": ["Limited capacity", "Spurious attractors", "Slow convergence"],
        },
        "MLP": {
            "name": "Multi-Layer Perceptron",
            "category": MethodCategory.NEURAL_NETWORK.value,
            "description": "Feedforward neural network with multiple hidden layers for supervised learning.",
            "best_for": ["classification", "regression", "function approximation", "non-linear fitting"],
            "strengths": ["Universal approximator", "Powerful", "Well-understood"],
            "weaknesses": ["Black-box", "Needs lots of data", "Sensitive to initialization"],
        },
        "Perceptron": {
            "name": "Single-Layer Perceptron",
            "category": MethodCategory.NEURAL_NETWORK.value,
            "description": "Single-layer linear classifier using Hebbian learning.",
            "best_for": ["linear classification", "binary classification", "simple patterns"],
            "strengths": ["Simple", "Fast", "Interpretable"],
            "weaknesses": ["Can only solve linear problems", "Single output", "Limited expressiveness"],
        },
        "SOM": {
            "name": "Self-Organizing Map",
            "category": MethodCategory.NEURAL_NETWORK.value,
            "description": "Unsupervised learning for dimensionality reduction and visualization.",
            "best_for": ["clustering", "visualization", "dimensionality reduction", "exploratory analysis"],
            "strengths": ["Interpretable visualization", "Topology preservation", "Unsupervised"],
            "weaknesses": ["Slow training", "Requires tuning neighborhood", "Limited for classification"],
        },
    }
    
    @staticmethod
    def build_system_prompt(available_methods: Dict[str, Dict[str, Any]]) -> str:
        """
        Builds the system prompt that instructs the LLM to act as a CI architect.
        
        Args:
            available_methods: Dictionary of method names to their PARAM_SPECS
            
        Returns:
            Complete system prompt string
        """
        methods_info = PromptBuilder._format_methods_info(available_methods)
        
        system_prompt = f"""You are an expert Computational Intelligence (CI) Architect with deep knowledge of neural networks, fuzzy systems, and evolutionary algorithms.

                            Your task is to SELECT THE BEST CI METHOD and its OPTIMAL PARAMETERS for a given problem.

                            === AVAILABLE CI METHODS & PARAMETER SPECIFICATIONS ===

                            {methods_info}

                            === YOUR DECISION FRAMEWORK ===

                            1. ANALYZE THE PROBLEM:
                            - Understand the problem type (optimization, classification, clustering, regression)
                            - Identify key characteristics (dimensionality, data size, time constraints, domain)
                            - Consider problem-specific challenges and requirements

                            2. METHOD SELECTION CRITERIA:
                            - Match problem characteristics to method strengths
                            - Consider computational budget (time, iterations)
                            - Evaluate scalability needs
                            - Account for problem structure (discrete vs continuous, linear vs non-linear)

                            3. PARAMETER TUNING STRATEGY:
                            - For EXPLORATION problems: Use larger populations, higher mutation/learning rates
                            - For EXPLOITATION problems: Use smaller populations, lower learning rates
                            - For HIGH-DIMENSIONAL problems: Increase mutation/exploration parameters
                            - For NOISY problems: Increase population/sample sizes
                            - For TIME-CONSTRAINED problems: Reduce iterations, increase population efficiency

                            4. CONFIDENCE & ALTERNATIVES:
                            - Provide confidence level (0.0-1.0) based on problem-method fit
                            - Suggest 2-3 alternative methods that could also work
                            - Include backup strategies for quick parameter adjustment

                            === PARAMETER NAMING CONVENTIONS ===

                            ACO: n_ants (swarm size), alpha (pheromone weight), beta (heuristic weight), evaporation_rate
                            GA: population_size, crossover_rate, mutation_rate, selection (tournament/roulette/rank)
                            GP: population_size, generations, max_depth, crossover_rate, mutation_rate
                            PSO: n_particles, w (inertia), c1, c2 (social/cognitive), velocity_clamp
                            FuzzyController: n_membership_functions, membership_type, defuzzification
                            Hopfield: max_iterations, threshold, async_update
                            MLP: hidden_layers (list), learning_rate, activation, max_epochs
                            Perceptron: learning_rate, max_epochs, bias
                            SOM: map_size (tuple), learning_rate_initial, max_epochs, topology

                            === RESPONSE FORMAT ===

                            You MUST respond with a valid JSON object following this structure:
                            {{
                                "selected_method": "<METHOD_NAME>",
                                "reasoning": "<Detailed explanation of why this method is best>",
                                "parameters": {{<key>: <value>, ...}},
                                "confidence": <0.0-1.0>,
                                "alternative_methods": ["<METHOD2>", "<METHOD3>"],
                                "expected_performance": "<low/medium/high>",
                                "warnings": [<any concerns>],
                                "backup_strategy": "<Optional alternative approach if performance is poor>"
                            }}

                            IMPORTANT: ALL parameters in the JSON must be valid for the selected method AND within the specified ranges.
                        """
        return system_prompt
    
    @staticmethod
    def _format_methods_info(available_methods: Dict[str, Dict[str, Any]]) -> str:
        formatted = []
        
        for method_name, param_specs in available_methods.items():
            if method_name not in PromptBuilder.METHOD_DESCRIPTIONS:
                continue
                
            desc = PromptBuilder.METHOD_DESCRIPTIONS[method_name]
            formatted.append(f"\n### {desc['name']} ({method_name})")
            formatted.append(f"Category: {desc['category']}")
            formatted.append(f"Description: {desc['description']}")
            formatted.append(f"Best for: {', '.join(desc['best_for'])}")
            formatted.append(f"Strengths: {', '.join(desc['strengths'])}")
            formatted.append(f"Weaknesses: {', '.join(desc['weaknesses'])}")
            formatted.append("\nParameters:")
            
            for param_name, spec in param_specs.items():
                formatted.append(f"  - {param_name}:")
                formatted.append(f"    Type: {spec.get('type', 'unknown').__name__ if hasattr(spec.get('type'), '__name__') else spec.get('type')}")
                
                if 'range' in spec:
                    formatted.append(f"    Range: {spec['range']}")
                if 'options' in spec:
                    formatted.append(f"    Options: {spec['options']}")
                if 'default' in spec:
                    formatted.append(f"    Default: {spec['default']}")
            
        return "\n".join(formatted)
    
    @staticmethod
    def build_problem_context(problem_info: Dict[str, Any]) -> str:
        """
        Builds a concise context string from problem information.
        
        Args:
            problem_info: Output from problem.get_info()
            
        Returns:
            Formatted problem context
        """
        context = [
            f"Problem Name: {problem_info.get('name', 'Unknown')}",
            f"Problem Type: {problem_info.get('type', 'Unknown')}",
        ]
        
        if 'dimension' in problem_info:
            context.append(f"Dimensionality: {problem_info['dimension']}")
        
        if 'metadata' in problem_info:
            meta = problem_info['metadata']
            if 'n_samples' in meta:
                context.append(f"Data Size: {meta['n_samples']} samples")
            if 'n_features' in meta:
                context.append(f"Features: {meta['n_features']}")
            if 'n_classes' in meta:
                context.append(f"Classes: {meta['n_classes']}")
            if 'optimization_type' in meta:
                context.append(f"Optimization Type: {meta['optimization_type']}")
        
        if 'bounds' in problem_info:
            context.append(f"Search Space: Bounded")
        
        if 'optimal_value' in problem_info:
            context.append(f"Known Optimum: {problem_info['optimal_value']}")
        
        return "\n".join(context)
    
    @staticmethod
    def build_feedback_prompt(
        problem_info: Dict[str, Any],
        previous_result: Dict[str, Any],
        previous_recommendation: Dict[str, Any]
    ) -> str:
        """
        Builds a feedback loop prompt to suggest parameter adjustments.
        
        Args:
            problem_info: Problem description
            previous_result: Results from previous execution
            previous_recommendation: Previous LLM recommendation
            
        Returns:
            Feedback prompt for next iteration
        """
        gap_percentage = previous_result.get('metrics', {}).get('gap_percentage', 'unknown')
        current_best = previous_result.get('best_fitness', 'unknown')
        method_used = previous_recommendation.get('selected_method', 'Unknown')
        params_used = previous_recommendation.get('parameters', {})
        
        feedback_prompt = f"""
                                FEEDBACK LOOP - PARAMETER TUNING ITERATION

                                Problem: {problem_info.get('name', 'Unknown')}
                                Previously Selected Method: {method_used}

                                === PREVIOUS EXECUTION RESULTS ===
                                Best Fitness: {current_best}
                                Gap to Optimal: {gap_percentage}%
                                Parameters Used: {json.dumps(params_used, indent=2)}

                                === ANALYSIS & ADJUSTMENT STRATEGY ===

                                Based on the execution results, determine if:
                                1. Gap is LARGE (> 20%): Try to INCREASE exploration
                                - Increase mutation/learning rates
                                - Increase population/swarm size
                                - Decrease elitism/exploitation parameters
                                
                                2. Gap is MODERATE (5-20%): Try to BALANCE exploration/exploitation
                                - Fine-tune rates moderately
                                - Consider different selection strategies
                                - Increase iterations if time allows

                                3. Gap is SMALL (< 5%): Try to REFINE solution
                                - Increase exploitation parameters
                                - Decrease learning rates (for gradient-based)
                                - Enable local search if available

                                PROVIDE an updated recommendation with:
                                - Whether to CONTINUE with same method or SWITCH
                                - Adjusted parameters with reasoning
                                - Updated confidence level
                                - Expected improvement from changes
                            """
        return feedback_prompt


def get_default_method_mapping() -> Dict[str, str]:
    return {
        "ACO": "src.methods.evolutionary.aco.AntColonyOptimization",
        "GA": "src.methods.evolutionary.ga.GeneticAlgorithm",
        "GP": "src.methods.evolutionary.gp.GeneticProgramming",
        "PSO": "src.methods.evolutionary.pso.PSO",
        "FuzzyController": "src.methods.fuzzy.controller.FuzzyController",
        "Hopfield": "src.methods.neural.hopfield.HopfieldNetwork",
        "MLP": "src.methods.neural.mlp.MLP",
        "Perceptron": "src.methods.neural.perceptron.Perceptron",
        "SOM": "src.methods.neural.som.SOM",
    }
