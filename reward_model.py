"""
Custom reward model for verl that uses GPT-5 judge.
This module provides a verl-compatible reward model interface.
"""

import os
import sys
from typing import List, Optional
import logging

# Ensure project root is on path when loaded by verl (e.g. from Ray workers)
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.judge import GPT5Judge

logger = logging.getLogger(__name__)

# # Ensure judge cost (INFO) is visible when run under verl if no other logging config
# if not logging.root.handlers and logging.root.level > logging.INFO:
#     logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


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
        total_cost_usd = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for prompt, response in zip(prompts, responses):
            try:
                # Extract seed question from prompt (remove template)
                seed_question = self._extract_seed_from_prompt(prompt)
                
                # Score with judge (returns dict with 'overall', 'usage', 'cost_usd')
                judge_scores = self.judge.score(seed_question, response)
                
                # Use the overall score as reward (judge already computes weighted average)
                reward = judge_scores.get("overall", 0.0)
                rewards.append(reward)
                # Aggregate judge cost and usage for logging
                total_cost_usd += judge_scores.get("cost_usd") or 0.0
                usage = judge_scores.get("usage")
                if usage:
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
            except Exception as e:
                logger.error(f"Error computing reward for prompt-response pair: {e}")
                rewards.append(0.0)  # Default to 0 on error

        if total_cost_usd > 0 or total_prompt_tokens > 0:
            logger.info(
                "Judge cost this batch: $%.6f USD (prompt_tokens=%d, completion_tokens=%d)",
                total_cost_usd,
                total_prompt_tokens,
                total_completion_tokens,
            )
        
        return rewards
    
    def _extract_seed_from_prompt(self, prompt: str) -> str:
        """
        Extract seed question from prompt template.

        Handles both a JSON-encoded chat list and a plain string.
        current_prompt ends with "Seed question: {seed_question}".
        """
        import json
        # Unwrap JSON chat list → get the user message content
        content = prompt
        try:
            if isinstance(prompt, str) and prompt.strip().startswith('['):
                prompt_list = json.loads(prompt)
                for msg in prompt_list:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        break
        except (json.JSONDecodeError, AttributeError):
            pass

        # Extract text after the last "Seed question:" marker
        marker = "Seed question:"
        idx = content.rfind(marker)
        if idx != -1:
            return content[idx + len(marker):].strip()

        # Fallback: return whatever we have
        return content.strip()


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
        cost_usd = judge_scores.get("cost_usd") or 0.0
        if cost_usd > 0:
            logger.info("Judge cost (single call): $%.6f USD", cost_usd)
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
