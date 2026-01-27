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
            prompt: Full prompt string (may include template)
            
        Returns:
            Seed question string
        """
        # Try to extract seed question from common prompt formats
        # This is a simple heuristic - adjust based on your prompt template
        
        # Look for "Given the following question:" pattern
        if "Given the following question:" in prompt:
            parts = prompt.split("Given the following question:")
            if len(parts) > 1:
                seed_part = parts[1].split("\n")[0].strip().strip('"').strip("'")
                if seed_part:
                    return seed_part
        
        # If no pattern found, assume prompt is just the seed question
        return prompt.strip()


# Factory function for verl to use
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
