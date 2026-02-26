"""Faithfulness & relevancy evaluation."""

import asyncio

from llama_index.core import Settings as LISettings
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.anthropic import Anthropic

from src.config import Settings


def _ensure_llm():
    """Make sure a global LLM is set for the evaluators."""
    if LISettings.llm is None:
        s = Settings()
        LISettings.llm = Anthropic(
            model=s.llm_model_name, api_key=s.anthropic_api_key
        )


def evaluate(query: str, response) -> dict:
    """Run faithfulness and relevancy evaluations. Returns a dict of results."""
    _ensure_llm()

    faithfulness_eval = FaithfulnessEvaluator()
    relevancy_eval = RelevancyEvaluator()

    loop = asyncio.get_event_loop()
    if loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()

    faith_result = asyncio.get_event_loop().run_until_complete(
        faithfulness_eval.aevaluate_response(response=response)
    )
    relev_result = asyncio.get_event_loop().run_until_complete(
        relevancy_eval.aevaluate_response(query=query, response=response)
    )

    return {
        "faithfulness": {
            "passing": faith_result.passing,
            "score": faith_result.score,
            "feedback": faith_result.feedback,
        },
        "relevancy": {
            "passing": relev_result.passing,
            "score": relev_result.score,
            "feedback": relev_result.feedback,
        },
    }
