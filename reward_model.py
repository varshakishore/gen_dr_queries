"""
Custom reward model for verl that uses GPT-5 judge.
This module provides a verl-compatible reward model interface.
"""

import os
from typing import List, Optional
import logging

from models.judge import GPT5Judge

logger = logging.getLogger(__name__)


class VerlRewardModel:
    """
    verl-compatible reward model that uses GPT-5 judge.
    
    This class implements the interface that verl expects for reward models.
    The judge already computes an overall score, so we use that directly.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        """
        Initialize reward model.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model_name: Judge model name
            base_url: Custom base URL for API
        """
        self.judge = GPT5Judge(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )
        logger.info("Initialized VerlRewardModel with GPT-5 judge")
    
    def __call__(
        self,
        prompts: List[str],
        responses: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for prompts and responses.
        
        This is the interface verl expects for reward models.
        
        Args:
            prompts: List of seed questions (prompts)
            responses: List of generated harder questions (responses)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of reward values (one per prompt-response pair)
        """
        rewards = []
        for prompt, response in zip(prompts, responses):
            try:
                # Extract seed question from prompt (remove template)
                seed_question = self._extract_seed_from_prompt(prompt)
                
                # Score with judge (returns dict with 'overall' score)
                judge_scores = self.judge.score(seed_question, response)
                
                # Use the overall score as reward (judge already computes weighted average)
                reward = judge_scores.get("overall", 0.0)
                rewards.append(reward)
            except Exception as e:
                logger.error(f"Error computing reward for prompt-response pair: {e}")
                rewards.append(0.0)  # Default to 0 on error
        
        return rewards
    
    def _extract_seed_from_prompt(self, prompt: str) -> str:
        """
        Extract seed question from prompt template.
        
        Args:
            prompt: Full prompt string (may include template or be a chat format)
            
        Returns:
            Seed question string
        """
        # Handle chat format (list of dicts) if prompt is a string representation
        import json
        try:
            # Try to parse as JSON if it's a string representation of a list
            if isinstance(prompt, str) and prompt.strip().startswith('['):
                prompt_list = json.loads(prompt)
                # Find the user message and extract question
                for msg in prompt_list:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        # Look for "Given the following question:" pattern
                        if "Given the following question:" in content or "generate a harder question" in content.lower():
                            # Extract the question after the instruction
                            parts = content.split("generate a harder question")
                            if len(parts) > 1:
                                question = parts[1].strip()
                                # Remove leading colons, newlines, and quotes
                                question = question.lstrip(":\n").strip().strip('"').strip("'")
                                if question:
                                    return question
                        # If no pattern, return the whole content
                        return content.strip()
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Try to extract seed question from common prompt formats (string format)
        # Look for "Given the following question:" pattern
        if "Given the following question:" in prompt:
            parts = prompt.split("Given the following question:")
            if len(parts) > 1:
                seed_part = parts[1].split("\n")[0].strip().strip('"').strip("'")
                if seed_part:
                    return seed_part
        
        # If no pattern found, assume prompt is just the seed question
        return prompt.strip()


# Global reward model instance for function-based API
_reward_model_instance: Optional[VerlRewardModel] = None


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> float:
    """
    VERL-compatible reward function signature.
    
    This function matches VERL's expected custom_reward_function signature:
    def compute_score(data_source, solution_str, ground_truth, extra_info=None)
    
    Args:
        data_source: Dataset identifier (ignored, but required by VERL)
        solution_str: Generated response (harder question)
        ground_truth: Original seed question (from data)
        extra_info: Optional additional information
        
    Returns:
        Reward score (float)
    """
    global _reward_model_instance
    
    # Initialize reward model if not already done
    if _reward_model_instance is None:
        _reward_model_instance = VerlRewardModel(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o",
        )
    
    try:
        # Use ground_truth as the seed question (prompt)
        # solution_str is the generated harder question (response)
        judge_scores = _reward_model_instance.judge.score(ground_truth, solution_str)
        reward = judge_scores.get("overall", 0.0)
        return reward
    except Exception as e:
        logger.error(f"Error computing reward: {e}")
        return 0.0


# Factory function for verl to use (class-based approach)
def create_reward_model(config: Optional[dict] = None) -> VerlRewardModel:
    """
    Factory function to create reward model from config.
    
    This can be called by verl to instantiate the reward model.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        VerlRewardModel instance
    """
    if config is None:
        config = {}
    
    return VerlRewardModel(
        api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
        model_name=config.get("model_name", "gpt-4o"),
        base_url=config.get("base_url"),
    )
