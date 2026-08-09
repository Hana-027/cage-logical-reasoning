from __future__ import annotations

from .schema import CounterfactualExample, Example

LABEL_INSTRUCTION = "Answer with exactly one of: true, false, unknown."


def direct_prompt(example: Example | CounterfactualExample) -> str:
    return f"""You are solving a logical reasoning problem.

{example.text}

{LABEL_INSTRUCTION}
Final answer:"""


def cot_prompt(example: Example | CounterfactualExample) -> str:
    return f"""You are solving a logical reasoning problem. Give a brief explanation, then answer.

{example.text}

Keep the explanation short and do not introduce facts not listed above.
{LABEL_INSTRUCTION}
"""


def cpa_prompt(example: Example | CounterfactualExample) -> str:
    return f"""You are solving a logical reasoning problem with numbered premises.

{example.text}

Return valid JSON with this schema:
{{
  "answer": "true | false | unknown",
  "causal_premises": ["premise ids such as F1 or R2 that are necessary for the answer"],
  "brief_explanation": "one short sentence"
}}

Use only premise IDs that appear in the problem. Do not include hidden reasoning.
"""


def cpa_counterfactual_prompt(source: Example, cf: CounterfactualExample, prior_premises: tuple[str, ...]) -> str:
    changed = ", ".join(cf.changed_ids)
    prior = ", ".join(prior_premises) if prior_premises else "none"
    return f"""You previously identified these causal premises for the original problem: {prior}.
The counterfactual intervention changed these premise IDs: {changed}.

Now solve the counterfactual problem from scratch:

{cf.text}

Return valid JSON with this schema:
{{
  "answer": "true | false | unknown",
  "causal_premises": ["premise ids necessary for the new answer"],
  "brief_explanation": "one short sentence"
}}
"""
