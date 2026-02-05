"""
MetaMind Agent - LLM-based CI method selector using Groq API.
Uses JSON mode to ensure structured outputs matching LLMRecommendation dataclass.
"""

import json
import logging
from typing import Dict, Any, Optional
from groq import Groq

from .schema import LLMRecommendationSchema
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
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        verbose: bool = True
    ):
        """
        Initialize the MetaMind Agent.
        
        Args:
            api_key: Groq API key
            model: Model identifier (default: llama-3.3-70b-versatile)
            temperature: LLM temperature for creativity vs consistency
            max_tokens: Maximum response tokens
            verbose: Enable logging
        """
        self.client = Groq(api_key=api_key)
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
            # Call Groq API with JSON mode
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}  # Enforce JSON output
            )
            
            # Extract and validate response
            response_text = response.choices[0].message.content
            
            if self.verbose:
                logger.debug(f"Raw LLM response: {response_text[:200]}...")
            
            # Parse and validate with Pydantic
            recommendation = LLMRecommendationSchema.model_validate_json(response_text)
            
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
            logger.error(f"Groq API error: {e}")
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
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": feedback_prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            recommendation = LLMRecommendationSchema.model_validate_json(response_text)
            
            if self.verbose:
                logger.info(f"Feedback recommendation: {recommendation.selected_method}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Feedback recommendation failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about agent usage."""
        return {
            "total_calls": self.call_count,
            "model": self.model,
            "temperature": self.temperature
        }