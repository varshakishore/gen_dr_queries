"""
Custom reward model for verl using DifficultyScorer.
Reward = difficulty(generated_question) / 10, normalized to [0, 1].
"""

import fcntl
import json
import logging
import os
import sys
from typing import Optional

# Ensure project root is on path when loaded by verl (e.g. from Ray workers)
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from estimate_question_difficulty import DifficultyScorer
from difficulty_signals import register_all_signals

logger = logging.getLogger(__name__)

_jsonl_path = os.path.join(_project_root, os.getenv("REWARD_JSONL_FILE", "logs/rewards.jsonl"))
os.makedirs(os.path.dirname(_jsonl_path), exist_ok=True)


def _write_jsonl(record: dict) -> None:
    """Append a record to the JSONL file. flock ensures safety across Ray worker processes."""
    with open(_jsonl_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _make_scorer(
    api_key: Optional[str] = None,
    model_name: str = "gpt-5-mini",
    base_url: Optional[str] = None,
) -> DifficultyScorer:
    scorer = DifficultyScorer(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        model_name=model_name,
        base_url=base_url,
    )
    register_all_signals(
        scorer,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        model_name=model_name,
        base_url=base_url,
        use_rubric=False,
    )
    return scorer


# Global scorer instance reused across compute_score calls
_scorer_instance: Optional[DifficultyScorer] = None


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> float:
    """
    VERL-compatible reward function.

    Args:
        data_source: Dataset identifier (required by VERL, unused here)
        solution_str: Generated harder question to score
        ground_truth: Original seed question (unused — we score the generated question directly)
        extra_info: Optional metadata

    Returns:
        Reward score in [0, 1]
    """
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = _make_scorer(model_name="gpt-5-mini")

    try:
        result = _scorer_instance.score(solution_str)
        reward = result["score"] / 10.0
        _write_jsonl({
            "response": solution_str,
            "ground_truth": ground_truth,
            "reward": reward,
            "score": result["score"],
            "reasoning": result.get("reasoning", ""),
            "context": result.get("context", {}),
            "cost_usd": result.get("cost_usd") or 0.0,
        })
        return reward
    except Exception as e:
        logger.error("Error computing reward: %s", e)
        return 0.0
