from __future__ import annotations

import json
from typing import Any

from .mpss import run_mpss_with_plan
from .schema import CounterfactualExample, Example, Prediction

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_strategy": {
            "type": "string",
            "enum": ["backward_chaining", "forward_expansion", "negated_query_search"],
        },
        "check_negation": {"type": "boolean"},
        "check_contradiction": {"type": "boolean"},
        "support_minimization": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["primary_strategy", "check_negation", "check_contradiction", "support_minimization", "reason"],
    "additionalProperties": False,
}


def strategy_prompt(example: Example | CounterfactualExample) -> str:
    return f"""You are a meta-cognitive planner for logical reasoning.

Your task is not to answer the question. Choose a proof-search strategy for a symbolic executor.

Available strategies:
- backward_chaining: best when the query likely follows from multi-hop rules.
- forward_expansion: best when many facts can derive useful intermediate conclusions.
- negated_query_search: best when the question may be false or unknown and the negated query should be prioritized.

Always decide whether the executor should check the negated query, check contradictions, and minimize support.

Problem:
{example.text}

Return only JSON matching the requested schema.
"""


def llm_guided_mpss_predict(example: Example | CounterfactualExample, plan: dict[str, Any], raw_plan: str = "") -> Prediction:
    result = run_mpss_with_plan(example.facts, example.rules, example.query, plan)
    trace = {
        "llm_plan": plan,
        "strategy": result.strategy,
        "expansions": result.expansions,
        "positive_found": result.positive_found,
        "negative_found": result.negative_found,
        "contradiction": result.contradiction,
        "raw_plan": raw_plan,
    }
    return Prediction(example.id, "llm_guided_mpss", result.label, result.support_ids, json.dumps(trace, sort_keys=True))


def fallback_plan() -> dict[str, Any]:
    return {
        "primary_strategy": "backward_chaining",
        "check_negation": True,
        "check_contradiction": True,
        "support_minimization": True,
        "reason": "Fallback deterministic plan when no LLM plan is available.",
    }
