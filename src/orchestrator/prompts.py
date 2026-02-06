import json
from typing import Dict, Any, List
from src.core.types import ProblemType, MethodCategory, OptimizationType


class PromptBuilder:
    
    METHOD_DESCRIPTIONS = {
        "ACO": {
            "name": "Ant Colony Optimization",
            "category": MethodCategory.SWARM_INTELLIGENCE.value,
            "description": "Bio-inspired metaheuristic for combinatorial optimization. Simulates pheromone trails.",
            "best_for": ["TSP", "routing", "scheduling", "combinatorial problems"],
            "problem_types": ["TSP", "combinatorial"],
            "strengths": ["Excellent for graph-based problems", "Fast convergence", "Distributed nature"],
            "weaknesses": ["Can get stuck in local optima", "Requires proper parameter tuning"],
        },
        "GA": {
            "name": "Genetic Algorithm",
            "category": MethodCategory.EVOLUTIONARY.value,
            "description": "Population-based evolutionary algorithm with 3 variants: (1) PERMUTATION for TSP/routing, (2) CONTINUOUS for real-valued optimization, (3) BINARY for boolean problems.",
            "best_for": ["TSP", "combinatorial optimization", "continuous optimization", "discrete optimization", "function optimization"],
            "problem_types": ["TSP", "combinatorial", "discrete", "continuous", "function_optimization"],
            "strengths": ["General-purpose", "Handles discrete and continuous problems", "Parallelizable", "Multiple variants available"],
            "weaknesses": ["Slow on high-dimensional problems", "Needs proper population sizing"],
            "ga_variants": {
                "permutation": "Use for TSP, routing, scheduling, job assignment - problems with permutations",
                "continuous": "Use for real-valued optimization, function optimization, continuous search spaces",
                "binary": "Use for binary/boolean problems, feature selection, knapsack problems"
            }
        },
        "GP": {
            "name": "Genetic Programming",
            "category": MethodCategory.EVOLUTIONARY.value,
            "description": "Evolves tree-based mathematical expressions for symbolic regression.",
            "best_for": ["symbolic regression", "function approximation", "expression discovery"],
            "problem_types": ["continuous", "regression"],
            "strengths": ["Interpretable results", "No need to specify form", "Automatic feature discovery"],
            "weaknesses": ["Slow", "Code bloat", "Requires large populations"],
        },
        "PSO": {
            "name": "Particle Swarm Optimization",
            "category": MethodCategory.SWARM_INTELLIGENCE.value,
            "description": "Simulates social behavior of bird flocking or fish schooling. ONLY for continuous optimization.",
            "best_for": ["continuous optimization", "function optimization", "multi-modal problems"],
            "problem_types": ["continuous", "function_optimization"],
            "strengths": ["Fast convergence", "Few parameters", "Good for continuous spaces"],
            "weaknesses": ["CANNOT handle discrete/combinatorial problems like TSP", "Can diverge"],
        },
        "FuzzyController": {
            "name": "Fuzzy Logic Controller",
            "category": MethodCategory.FUZZY_SYSTEM.value,
            "description": "Uses fuzzy sets and rules for classification and control.",
            "best_for": ["classification", "control systems", "time series", "decision support"],
            "problem_types": ["classification"],
            "strengths": ["Interpretable rules", "Handles imprecision", "Robust"],
            "weaknesses": ["Requires domain knowledge", "Manual rule definition", "Not best for pure prediction"],
        },
        "Hopfield": {
            "name": "Hopfield Network",
            "category": MethodCategory.NEURAL_NETWORK.value,
            "description": "Recurrent neural network for pattern completion and associative memory.",
            "best_for": ["pattern recognition", "pattern completion", "memory retrieval", "optimization"],
            "problem_types": ["pattern_recognition", "continuous"],
            "strengths": ["Associative memory", "Pattern completion", "Energy-based stability"],
            "weaknesses": ["Limited capacity", "Spurious attractors", "Slow convergence"],
        },
        "MLP": {
            "name": "Multi-Layer Perceptron",
            "category": MethodCategory.NEURAL_NETWORK.value,
            "description": "Feedforward neural network with multiple hidden layers for supervised learning.",
            "best_for": ["classification", "regression", "function approximation", "non-linear fitting"],
            "problem_types": ["classification", "regression"],
            "strengths": ["Universal approximator", "Powerful", "Well-understood"],
            "weaknesses": ["Black-box", "Needs lots of data", "Sensitive to initialization"],
        },
        "Perceptron": {
            "name": "Single-Layer Perceptron",
            "category": MethodCategory.NEURAL_NETWORK.value,
            "description": "Single-layer linear classifier using Hebbian learning.",
            "best_for": ["linear classification", "binary classification", "simple patterns"],
            "problem_types": ["classification"],
            "strengths": ["Simple", "Fast", "Interpretable"],
            "weaknesses": ["Can only solve linear problems", "Single output", "Limited expressiveness"],
        },
        "SOM": {
            "name": "Self-Organizing Map",
            "category": MethodCategory.NEURAL_NETWORK.value,
            "description": "Unsupervised learning for dimensionality reduction and visualization.",
            "best_for": ["clustering", "visualization", "dimensionality reduction", "exploratory analysis"],
            "problem_types": ["clustering"],
            "strengths": ["Interpretable visualization", "Topology preservation", "Unsupervised"],
            "weaknesses": ["Slow training", "Requires tuning neighborhood", "Limited for classification"],
        },
    }
    
    @staticmethod
    def build_system_prompt(available_methods: Dict[str, Dict[str, Any]]) -> str:

        methods_info = PromptBuilder._format_methods_info(available_methods)
        
        system_prompt = f"""You are an expert Computational Intelligence (CI) Architect with deep knowledge of neural networks, fuzzy systems, and evolutionary algorithms.

                            Your task is to SELECT THE BEST CI METHOD and its OPTIMAL PARAMETERS for a given problem.

                            === AVAILABLE CI METHODS & PARAMETER SPECIFICATIONS ===

                            {methods_info}

                            === YOUR DECISION FRAMEWORK ===

                            1. ANALYZE THE PROBLEM:
                            - Understand the problem type (optimization, classification, clustering, regression)
                            - Identify if it's TSP/combinatorial OR continuous optimization OR classification/clustering
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
                            GA: population_size, crossover_rate, mutation_rate, selection (tournament/roulette/rank), ga_type (permutation|continuous|binary), crossover_type (depends on ga_type)
                            GP: population_size, generations, max_depth, crossover_rate, mutation_rate
                            PSO: n_particles, w (inertia), c1, c2 (social/cognitive), velocity_clamp
                            FuzzyController: n_membership_functions, membership_type, defuzzification
                            Hopfield: max_iterations, threshold, async_update
                            MLP: hidden_layers (list), learning_rate, activation, max_epochs
                            Perceptron: learning_rate, max_epochs, bias
                            SOM: map_size (tuple), learning_rate_initial, max_epochs, topology

                            === GA TYPE & CROSSOVER TYPE SELECTION ===

                            When recommending GA:
                            1. IF problem is TSP/routing/scheduling/combinatorial → ga_type: "permutation", crossover_type: "pmx" or "ox" or "cx"
                            2. IF problem is continuous optimization/function optimization → ga_type: "continuous", crossover_type: "single_point" or "two_point", include "bounds" parameter
                            3. IF problem is binary/boolean feature selection → ga_type: "binary", crossover_type: "single_point" or "two_point"
                            
                            For continuous GA: can optionally include gaussian_std parameter (0.01-1.0, typical: 0.1-0.2)

                            === RESPONSE FORMAT ===

                            You MUST respond with a valid JSON object following this structure:
                            {{
                                "selected_method": "<METHOD_IDENTIFIER>",
                                "reasoning": "<Detailed explanation of why this method is best>",
                                "parameters": {{<key>: <value>, ...}},
                                "confidence": <0.0-1.0>,
                                "alternative_methods": ["<METHOD_ID2>", "<METHOD_ID3>"],
                                "expected_performance": "<low|medium|high>",
                                "warnings": [<any concerns>],
                                "backup_strategy": "<Optional alternative approach if performance is poor>"
                            }}

                            CRITICAL REQUIREMENTS:
                            - Use the SHORT METHOD IDENTIFIER (e.g., "ACO", "GA", "PSO", "MLP") NOT the full name
                            - The method identifier MUST match EXACTLY what is shown after "METHOD IDENTIFIER:" above
                            - "expected_performance" MUST be EXACTLY one of: "low", "medium", or "high" (no hyphens, no combinations)
                            - ALL string values must use proper JSON escaping (escape quotes and newlines)
                            - ALL parameters must be valid for the selected method AND within the specified ranges
                            - Keep explanations concise to avoid JSON formatting issues
                        """
        return system_prompt
    
    @staticmethod
    def _format_methods_info(available_methods: Dict[str, Dict[str, Any]]) -> str:
        formatted = []
        
        for method_name, param_specs in available_methods.items():
            if method_name not in PromptBuilder.METHOD_DESCRIPTIONS:
                continue
                
            desc = PromptBuilder.METHOD_DESCRIPTIONS[method_name]
            formatted.append(f"\n### METHOD IDENTIFIER: '{method_name}' - {desc['name']}")
            formatted.append(f"Category: {desc['category']}")
            formatted.append(f"Description: {desc['description']}")
            if 'problem_types' in desc:
                formatted.append(f"**Supported Problem Types: {', '.join(desc['problem_types'])}**")
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

                                === REQUIRED RESPONSE FORMAT ===

                                You MUST respond with a valid JSON object following the EXACT same structure as before:
                                {{
                                    "selected_method": "<METHOD_IDENTIFIER>",
                                    "reasoning": "<Detailed explanation of parameter adjustments>",
                                    "parameters": {{<adjusted_parameters>}},
                                    "confidence": <0.0-1.0>,
                                    "alternative_methods": ["<alternatives>"],
                                    "expected_performance": "<low|medium|high>",
                                    "warnings": ["<any concerns>"],
                                    "backup_strategy": "<Optional fallback>"
                                }}

                                CRITICAL REQUIREMENTS: 
                                - Use SHORT METHOD IDENTIFIER (e.g., "{method_used}") NOT the full method name
                                - For "selected_method": Use "{method_used}" if continuing, or choose a different method identifier if switching
                                - For "expected_performance": MUST be exactly "low", "medium", or "high"
                                - Provide adjusted parameters with clear reasoning for each change
                            """
        return feedback_prompt
    
    @staticmethod
    def build_multi_method_prompt(
        problem_info: Dict[str, Any],
        available_methods: Dict[str, Dict[str, Any]],
        num_methods: int = 3
    ) -> str:
        problem_context = PromptBuilder.build_problem_context(problem_info)
        
        prompt = f"""
                    === MULTI-METHOD ORCHESTRATION MODE ===
                    
                    Instead of selecting ONE method, you will select {num_methods} DIFFERENT methods to run in parallel.
                    This allows us to compare multiple approaches and determine which works best for this problem.
                    
                    === PROBLEM TO SOLVE ===
                    {problem_context}
                    
                    === YOUR TASK ===
                    
                    Select {num_methods} different CI methods that:
                    1. Represent DIVERSE approaches (e.g., don't pick GA and GP together - too similar)
                    2. Are ALL potentially suitable for this problem type
                    3. Have complementary strengths (exploration vs exploitation, speed vs accuracy, etc.)
                    
                    For EACH selected method, provide optimal parameters based on:
                    - Problem characteristics (size, dimensionality, constraints)
                    - Method strengths and typical use cases
                    - Computational budget (keep iterations reasonable)
                    
                    === GA VARIANTS REMINDER ===
                    When including GA in your selection, choose the appropriate variant:
                    - ga_type: "permutation" with crossover_type from [pmx, ox, cx] for TSP/combinatorial
                    - ga_type: "continuous" with crossover_type from [single_point, two_point] for continuous optimization
                    - ga_type: "binary" with crossover_type from [single_point, two_point] for binary problems
                    
                    === RESPONSE FORMAT ===
                    
                    You MUST respond with valid JSON:
                    {{
                        "selected_methods": ["<METHOD_ID1>", "<METHOD_ID2>", "<METHOD_ID3>", ...],
                        "reasoning": "<Why these methods were chosen for comparison>",
                        "method_parameters": {{
                            "<METHOD_ID1>": {{<parameters_dict>}},
                            "<METHOD_ID2>": {{<parameters_dict>}},
                            ...
                        }},
                        "confidence": <0.0-1.0>,
                        "comparison_criteria": ["best_fitness", "execution_time", "convergence_speed"],
                        "expected_best_method": "<METHOD_ID or null>"
                    }}
                    
                    CRITICAL REQUIREMENTS:
                    - Use SHORT METHOD IDENTIFIERS (e.g., "ACO", "GA", "PSO") NOT full names like "Ant Colony Optimization"
                    - The identifiers are shown after "METHOD IDENTIFIER:" in the available methods list above
                    - Select EXACTLY {num_methods} methods
                    - All methods must be from the available methods list
                    - Provide complete, valid parameters for EACH method
                    - Keep reasoning concise to avoid JSON formatting issues
                """
        return prompt
    
    @staticmethod
    def build_multi_result_analysis_prompt(
        problem_info: Dict[str, Any],
        execution_results: Dict[str, Dict[str, Any]]
    ) -> str:
        problem_context = PromptBuilder.build_problem_context(problem_info)
        
        results_summary = []
        for method_name, result in execution_results.items():
            summary = f"""
            Method: {method_name}
            - Best Fitness: {result.get('best_fitness', 'N/A')}
            - Execution Time: {result.get('execution_time', 'N/A'):.2f}s
            - Iterations: {result.get('iterations', 'N/A')}
            - Gap from Optimal: {result.get('metrics', {}).get('gap_percentage', 'N/A')}%
            - Success: {result.get('success', True)}
            """
            results_summary.append(summary)
        
        results_text = "\n".join(results_summary)
        
        prompt = f"""
                    === MULTI-METHOD RESULT ANALYSIS ===
                    
                    You have executed multiple CI methods on the same problem. Now analyze the results and recommend the BEST method.
                    
                    === PROBLEM INFORMATION ===
                    {problem_context}
                    
                    === EXECUTION RESULTS ===
                    {results_text}
                    
                    === YOUR ANALYSIS TASK ===
                    
                    1. **Compare Performance**: Which method achieved the best fitness? Consider:
                       - Solution quality (fitness value, gap from optimal)
                       - Convergence speed (how quickly it found good solutions)
                       - Computational efficiency (execution time)
                       - Reliability (did it consistently find good solutions)
                    
                    2. **Rank Methods**: Order all methods from best to worst based on overall performance
                    
                    3. **Provide Detailed Analysis**: Explain:
                       - Why the recommended method performed best
                       - Strengths and weaknesses of each method
                       - Trade-offs between methods (speed vs accuracy, etc.)
                    
                    4. **Suggest Next Steps**: What should be done to further improve results?
                       - Parameter tuning for the best method
                       - Hybrid approaches combining strengths
                       - Additional methods to try
                    
                    === RESPONSE FORMAT ===
                    
                    You MUST respond with valid JSON:
                    {{
                        "recommended_method": "<BEST_METHOD_NAME>",
                        "ranking": ["<METHOD1>", "<METHOD2>", "<METHOD3>", ...],
                        "analysis": "<Detailed comparison and explanation>",
                        "performance_comparison": {{
                            "<METHOD1>": "<Brief performance summary>",
                            "<METHOD2>": "<Brief performance summary>",
                            ...
                        }},
                        "confidence": <0.0-1.0>,
                        "next_steps": ["<step1>", "<step2>", ...]
                    }}
                    
                    IMPORTANT:
                    - Be objective in your analysis
                    - Consider multiple criteria, not just fitness
                    - Keep text concise to avoid JSON formatting issues
                    - Provide actionable next steps
                """
        return prompt


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
