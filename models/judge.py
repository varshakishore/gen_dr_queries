"""
Minimal GPT5Judge: scores (seed_question, generated_question) using an LLM.
Returns a dict with "overall" (float 0-1), "usage" (token counts), and "cost_usd" (approx). Used by reward_model.py.
"""

import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Approximate USD per 1M tokens (input, output) for common models (used for cost logging)
_DEFAULT_COST_PER_1M = (2.50, 10.0)  # gpt-4o: input, output
_MODEL_COST_PER_1M = {
    "gpt-4.1": (2, 8),
    "gpt-5-mini": (0.25, 2),
    "gpt-5-nano": (0.05, 0.4),
}


class GPT5Judge:
    """
    Judge that scores how much harder a generated question is compared to a seed question.
    Uses OpenAI-compatible API (model_name, api_key, base_url).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5-nano",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except Exception as e:
                logger.warning("OpenAI client not available: %s. Judge will return 0.5.", e)
        return self._client

    def _cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Approximate cost in USD from token counts using model pricing."""
        in_1m, out_1m = _MODEL_COST_PER_1M.get(
            self.model_name, _DEFAULT_COST_PER_1M
        )
        return (prompt_tokens / 1e6 * in_1m) + (completion_tokens / 1e6 * out_1m)

    def score(self, seed_question: str, generated_question: str) -> dict:
        """
        Score how much harder the generated question is (0-1).
        Returns dict with "overall" (float), "usage" (dict with token counts), and "cost_usd" (float).
        """
        client = self._get_client()
        if client is None:
            return {"overall": 0.5, "usage": None, "cost_usd": 0.0}

        prompt = f"""You are a judge for question difficulty.

Seed question: {seed_question!r}

Generated (harder) question: {generated_question!r}

Rate how much harder the generated question is compared to the seed, from 0 (not harder) to 10 (much harder and still valid).
Reply with a single number between 0 and 10 (e.g. 7)."""

        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )
            usage = getattr(resp, "usage", None)
            usage_dict = None
            cost_usd = 0.0
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                usage_dict = {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": getattr(usage, "total_tokens", pt + ct) or (pt + ct),
                }
                cost_usd = self._cost_usd(pt, ct)

            text = (resp.choices[0].message.content or "").strip()
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            if match:
                raw = float(match.group(1))
                overall = min(1.0, max(0.0, raw / 10.0))
                logger.warning("jcost=%.6f", cost_usd)
                return {"overall": overall, "usage": usage_dict, "cost_usd": cost_usd}
            return {"overall": 0.5, "usage": usage_dict, "cost_usd": cost_usd}
        except Exception as e:
            logger.warning("Judge API call failed: %s. Returning 0.5.", e)
        return {"overall": 0.5, "usage": None, "cost_usd": 0.0}
