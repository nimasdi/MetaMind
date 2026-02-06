"""
MetaMind Agent - LLM-based CI method selector using Groq API.
Uses JSON mode to ensure structured outputs matching LLMRecommendation dataclass.
"""

import json
import logging
from typing import Dict, Any, Optional
import openai
from openai import OpenAI

from .schema import LLMRecommendationSchema, MultiMethodRecommendationSchema, MultiMethodResultAnalysisSchema
from .prompts import PromptBuilder


logger = logging.getLogger(__name__)


class MetaMindAgent:
    """
    Intelligent agent that uses Groq's LLM to select and configure CI methods.
    Leverages JSON mode for structured outputs and Pydantic validation.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        verbose: bool = True
    ):
        """
        Initialize the MetaMind Agent.
        
        Args:
            api_key: DeepInfra API key
            model: Model identifier (default: meta-llama/Meta-Llama-3.1-70B-Instruct)
            temperature: LLM temperature for creativity vs consistency
            max_tokens: Maximum response tokens
            verbose: Enable logging
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.call_count = 0
        
        if verbose:
            logger.info(f"MetaMindAgent initialized with model: {model}")
    
    def get_recommendation(
        self,
        problem_info: Dict[str, Any],
        available_methods: Dict[str, Dict[str, Any]],
        context: Optional[str] = None
    ) -> LLMRecommendationSchema:
        """
        Get LLM recommendation for method selection and parameter configuration.
        
        Args:
            problem_info: Problem metadata from problem.get_info()
            available_methods: Dictionary mapping method names to PARAM_SPECS
            context: Optional additional context for the LLM
            
        Returns:
            LLMRecommendationSchema with selected method and parameters
            
        Raises:
            ValueError: If response cannot be parsed or validation fails
            Exception: If Groq API call fails
        """
        self.call_count += 1
        
        # Build system and user prompts
        system_prompt = PromptBuilder.build_system_prompt(available_methods)
        problem_context = PromptBuilder.build_problem_context(problem_info)
        
        user_prompt = f"""
                            === PROBLEM TO SOLVE ===
                            {problem_context}

                            {f"Additional Context: {context}" if context else ""}

                            Please analyze this problem and recommend the BEST CI method with optimal parameters.
                            Return your response as valid JSON.
                        """
        
        if self.verbose:
            logger.info(f"[Call #{self.call_count}] Requesting recommendation for: {problem_info.get('name', 'Unknown')}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            
            response_text = response.choices[0].message.content
            
            if self.verbose:
                logger.debug(f"Raw LLM response length: {len(response_text)} chars")
                print(f"\n{'='*60}\nLLM MULTI-METHOD RECOMMENDATION:\n{'='*60}\n{response_text}\n{'='*60}\n")
                logger.debug(f"Raw LLM response: {response_text[:200]}...")
            
            # Check for incomplete JSON (common signs of truncation)
            if not response_text.strip().endswith('}'):
                logger.warning("Response appears truncated (doesn't end with '}')")
                logger.error(f"Full truncated response:\n{response_text}")
                raise ValueError(
                    f"LLM response appears truncated. Consider increasing max_tokens (currently {self.max_tokens}). "
                    f"Response length: {len(response_text)} chars"
                )
            
            try:
                recommendation = LLMRecommendationSchema.model_validate_json(response_text)
            except Exception as parse_error:
                logger.error(f"Failed to parse/validate JSON response: {parse_error}")
                logger.error(f"Full response text:\n{response_text}")
                raise
            
            if self.verbose:
                logger.info(
                    f"Recommendation: {recommendation.selected_method} "
                    f"(confidence: {recommendation.confidence:.2f})"
                )
            
            return recommendation
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"LLM did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"API error: {e}")
            raise
    
    def get_feedback_recommendation(
        self,
        problem_info: Dict[str, Any],
        available_methods: Dict[str, Dict[str, Any]],
        previous_result: Dict[str, Any],
        previous_recommendation: Dict[str, Any]
    ) -> LLMRecommendationSchema:
        """
        Get LLM feedback for parameter tuning based on previous execution.
        Implements the feedback loop for iterative improvement.
        
        Args:
            problem_info: Problem metadata
            available_methods: Available methods and their specs
            previous_result: Results from previous execution
            previous_recommendation: Previous LLM recommendation
            
        Returns:
            Updated LLMRecommendationSchema with adjusted parameters
        """
        system_prompt = PromptBuilder.build_system_prompt(available_methods)
        feedback_prompt = PromptBuilder.build_feedback_prompt(
            problem_info,
            previous_result,
            previous_recommendation
        )
        
        if self.verbose:
            logger.info(f"[Call #{self.call_count + 1}] Requesting feedback adjustment")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": feedback_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            
            response_text = response.choices[0].message.content
            recommendation = LLMRecommendationSchema.model_validate_json(response_text)
            
            if self.verbose:
                logger.info(f"Feedback recommendation: {recommendation.selected_method}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Feedback recommendation failed: {e}")
            raise
    
    def interpret_results(
        self,
        problem_info: Dict[str, Any],
        execution_result: Dict[str, Any],
        recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Step 6: Interpret execution results and provide analysis.
        Analyzes performance, compares with expected, explains results,
        and provides improvement recommendations.
        
        Args:
            problem_info: Problem information from problem.get_info()
            execution_result: Execution result metrics
            recommendation: The LLM recommendation that was used
            
        Returns:
            Dictionary with interpretation including:
            - performance_assessment: 'excellent'/'good'/'acceptable'/'poor'
            - performance_explanation: Natural language explanation
            - comparison_with_expected: How actual compares to expected
            - improvement_recommendations: List of tuples (type, suggestion)
            - confidence_assessment: Confidence in the solution
            - next_steps: Suggested next steps
        """
        prompt = f"""
You are an expert in computational intelligence methods. Analyze the following execution results and provide a comprehensive interpretation.

## Problem Information:
{json.dumps(problem_info, indent=2)}

## Execution Result:
- Method used: {recommendation.get('selected_method')}
- Parameters: {json.dumps(recommendation.get('parameters', {}), indent=2)}
- Best fitness: {execution_result.get('best_fitness')}
- Expected performance: {recommendation.get('expected_performance')}
- Computation time: {execution_result.get('execution_time'):.2f}s
- Iterations: {execution_result.get('iterations')}
- Gap from optimal: {execution_result.get('metrics', {}).get('gap_percentage', 'N/A')}%

## Your Analysis Task:
Provide a structured analysis with:

1. **Performance Assessment**: Rate as 'excellent' (gap < 1%), 'good' (gap 1-5%), 'acceptable' (gap 5-15%), or 'poor' (gap > 15%)

2. **Performance Explanation**: Explain the results in natural language:
   - Was convergence smooth or erratic?
   - Did the method find the solution quickly or slowly?
   - Is the computation time reasonable?

3. **Comparison with Expected**: How does actual performance compare to the expected performance?

4. **Improvement Recommendations**: Provide specific, actionable suggestions:
   - Parameter tuning suggestions (e.g., "increase population_size to 150")
   - Alternative method recommendations (e.g., "try GA for better exploration")
   - Hybrid approach suggestions (e.g., "combine with local search")

5. **Confidence Assessment**: Rate confidence as 'HIGH' (solution is reliable), 'MEDIUM' (acceptable but could improve), or 'LOW' (needs improvement)

6. **Next Steps**: What should be tried next?

Format your response as VALID JSON with these exact keys:
{{
    "performance_assessment": "good|excellent|acceptable|poor",
    "performance_explanation": "...",
    "comparison_with_expected": "...",
    "improvement_recommendations": [
        {{"type": "parameter_tuning|alternative_method|hybrid_approach", "suggestion": "..."}},
        ...
    ],
    "confidence_assessment": "HIGH|MEDIUM|LOW",
    "next_steps": ["step1", "step2", ...]
}}

IMPORTANT: Keep all text fields CONCISE and use proper JSON escaping. Avoid quotes within strings or use escaped quotes (\\")
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            
            response_text = response.choices[0].message.content
            
            self.call_count += 1
            
            # Parse the response with better error handling
            try:
                interpretation = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error: {json_err}")
                logger.debug(f"Raw response text: {response_text[:500]}...")
                # Try to extract JSON if it's wrapped in markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                    interpretation = json.loads(response_text)
                else:
                    raise json_err
            
            if self.verbose:
                logger.info(
                    f"Result interpretation: {interpretation.get('performance_assessment')} "
                    f"({interpretation.get('confidence_assessment')} confidence)"
                )
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Result interpretation failed: {e}")
            raise
    
    def get_multi_method_recommendation(
        self,
        problem_info: Dict[str, Any],
        available_methods: Dict[str, Dict[str, Any]],
        num_methods: int = 3,
        context: Optional[str] = None
    ) -> MultiMethodRecommendationSchema:
        self.call_count += 1
        
        if num_methods < 2 or num_methods > 5:
            raise ValueError("num_methods must be between 2 and 5")
        
        system_prompt = PromptBuilder.build_system_prompt(available_methods)
        multi_method_prompt = PromptBuilder.build_multi_method_prompt(
            problem_info, 
            available_methods, 
            num_methods
        )
        
        if context:
            multi_method_prompt += f"\n\nAdditional Context: {context}"
        
        if self.verbose:
            logger.info(
                f"[Call #{self.call_count}] Requesting {num_methods} methods for: "
                f"{problem_info.get('name', 'Unknown')}"
            )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": multi_method_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            
            response_text = response.choices[0].message.content
            
            if self.verbose:
                logger.debug(f"Raw LLM response length: {len(response_text)} chars")
            
            # Check for truncation
            if not response_text.strip().endswith('}'):
                logger.warning("Response appears truncated")
                raise ValueError(
                    f"LLM response appears truncated. Consider increasing max_tokens. "
                    f"Response length: {len(response_text)} chars"
                )
            
            try:
                recommendation = MultiMethodRecommendationSchema.model_validate_json(response_text)
            except Exception as parse_error:
                logger.error(f"Failed to parse/validate JSON response: {parse_error}")
                logger.error(f"Full response text:\n{response_text}")
                raise
            
            if self.verbose:
                logger.info(
                    f"Multi-method recommendation: {', '.join(recommendation.selected_methods)} "
                    f"(confidence: {recommendation.confidence:.2f})"
                )
            
            return recommendation
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"LLM did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"API error: {e}")
            raise
    
    def analyze_multi_method_results(
        self,
        problem_info: Dict[str, Any],
        execution_results: Dict[str, Dict[str, Any]]
    ) -> MultiMethodResultAnalysisSchema:
        """
        Analyze results from multiple method executions and recommend the best.
        This is the final step in multi-method orchestration.
        
        Args:
            problem_info: Problem metadata
            execution_results: Dictionary mapping method names to execution results
                             Each result should contain: best_fitness, execution_time, 
                             iterations, metrics, success
            
        Returns:
            MultiMethodResultAnalysisSchema with recommended method and analysis
            
        Raises:
            ValueError: If response cannot be parsed or validation fails
            Exception: If API call fails
        """
        self.call_count += 1
        
        # Build analysis prompt
        analysis_prompt = PromptBuilder.build_multi_result_analysis_prompt(
            problem_info,
            execution_results
        )
        
        if self.verbose:
            logger.info(
                f"[Call #{self.call_count}] Analyzing results from "
                f"{len(execution_results)} methods"
            )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            
            response_text = response.choices[0].message.content
            
            if self.verbose:
                logger.debug(f"Raw analysis response length: {len(response_text)} chars")
                print(f"\n{'='*60}\nLLM RESULT ANALYSIS:\n{'='*60}\n{response_text}\n{'='*60}\n")
            
            # Check for truncation
            if not response_text.strip().endswith('}'):
                logger.warning("Response appears truncated")
                raise ValueError(
                    f"LLM response appears truncated. Response length: {len(response_text)} chars"
                )
            
            try:
                analysis = MultiMethodResultAnalysisSchema.model_validate_json(response_text)
            except Exception as parse_error:
                logger.error(f"Failed to parse/validate JSON response: {parse_error}")
                logger.error(f"Full response text:\n{response_text}")
                raise
            
            if self.verbose:
                logger.info(
                    f"Recommended method: {analysis.recommended_method} "
                    f"(confidence: {analysis.confidence:.2f})"
                )
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"LLM did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"API error: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self.call_count,
            "model": self.model,
            "temperature": self.temperature
        }