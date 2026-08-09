from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .counterfactuals import generate_counterfactuals
from .prompts import cpa_prompt
from .schema import Example, Prediction


@dataclass(frozen=True)
class CAGEConfig:
    max_counterfactuals: int = 4
    max_tokens: int = 768


def cage_predict(
    example: Example,
    call_llm: Callable[[str, int, bool, dict | None], str],
    extract_label: Callable[[str], str],
    extract_premises: Callable[[str], tuple[str, ...]],
    config: CAGEConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or CAGEConfig()
    raw_rows: list[dict[str, str]] = []

    initial_prompt = cpa_prompt(example)
    initial_raw = call_llm(initial_prompt, config.max_tokens, True, None)
    initial_label = extract_label(initial_raw)
    initial_premises = extract_premises(initial_raw)
    raw_rows.append({"example_id": example.id, "method": "llm_cage_initial", "raw_response": initial_raw})

    diagnostics = _diagnose(example, initial_label, initial_premises, call_llm, extract_label, config)
    repair_prompt = _repair_prompt(example, initial_raw, initial_label, initial_premises, diagnostics)
    repaired_raw = call_llm(repair_prompt, config.max_tokens, True, None)
    repaired_label = extract_label(repaired_raw)
    repaired_premises = extract_premises(repaired_raw)
    raw_rows.append({"example_id": example.id, "method": "llm_cage", "raw_response": repaired_raw})

    trace = {
        "initial_answer": initial_label,
        "initial_premises": list(initial_premises),
        "diagnostics": diagnostics,
        "repaired_raw": _safe_json(repaired_raw),
    }
    return Prediction(example.id, "llm_cage", repaired_label, repaired_premises, json.dumps(trace, sort_keys=True)), raw_rows


def _diagnose(
    example: Example,
    initial_label: str,
    initial_premises: tuple[str, ...],
    call_llm: Callable[[str, int, bool, dict | None], str],
    extract_label: Callable[[str], str],
    config: CAGEConfig,
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    cfs = generate_counterfactuals([example], seed=17, max_per_example=config.max_counterfactuals)
    claimed = set(initial_premises)
    for cf in cfs:
        cf_prompt = _counterfactual_check_prompt(example, cf, initial_label, initial_premises)
        cf_raw = call_llm(cf_prompt, min(config.max_tokens, 512), True, None)
        cf_pred = extract_label(cf_raw)
        failure = _classify_failure(cf, initial_label, cf_pred, claimed)
        diagnostics.append(
            {
                "counterfactual_id": cf.id,
                "intervention_type": cf.intervention_type,
                "cf_family": cf.cf_family,
                "expected_relation": cf.expected_relation,
                "changed_ids": ",".join(cf.changed_ids),
                "expected_answer": cf.label,
                "counterfactual_answer": cf_pred,
                "failure_type": failure,
            }
        )
    return diagnostics


def _classify_failure(cf, initial_label: str, cf_pred: str, claimed: set[str]) -> str:
    changed = set(cf.changed_ids)
    if cf.conflict_label or cf.cf_family == "contradiction_injection":
        return "missed_contradiction" if cf_pred not in {"unknown", "ambiguous"} else "ok"
    if cf.expected_relation == "changed":
        if cf_pred == initial_label:
            return "insensitive_to_causal_change" if changed & claimed else "missed_proof_break"
        return "ok"
    if cf.expected_relation == "preserved":
        if cf_pred != initial_label:
            return "distractor_sensitive" if not (changed & claimed) else "unstable_support"
        return "ok"
    return "ok"


def _counterfactual_check_prompt(example, cf, initial_label: str, initial_premises: tuple[str, ...]) -> str:
    return f"""You are checking whether a logical answer is causally faithful.

Original answer: {initial_label}
Claimed causal premises: {', '.join(initial_premises) if initial_premises else 'none'}
Counterfactual intervention: changed premise ids {', '.join(cf.changed_ids)}
Expected relation to original answer: {cf.expected_relation}

Counterfactual problem:
{cf.text}

Return valid JSON only:
{{
  "answer": "true | false | unknown",
  "brief_explanation": "one short sentence explaining the counterfactual answer"
}}
"""


def _repair_prompt(example: Example, initial_raw: str, initial_label: str, initial_premises: tuple[str, ...], diagnostics: list[dict[str, str]]) -> str:
    failed = [d for d in diagnostics if d["failure_type"] != "ok"]
    diagnostic_text = "\n".join(
        f"- {d['failure_type']}: changed {d['changed_ids']} in {d['cf_family']}; expected {d['expected_relation']}; counterfactual answer was {d['counterfactual_answer']}"
        for d in failed
    ) or "- No failed counterfactual checks were found; verify the original answer and causal premises one more time."
    return f"""You are repairing a logical reasoning answer using counterfactual causal attribution feedback.

Problem:
{example.text}

Initial model output:
{initial_raw}

Initial answer: {initial_label}
Initial causal premises: {', '.join(initial_premises) if initial_premises else 'none'}

Counterfactual diagnostics:
{diagnostic_text}

Revise the answer and causal premises. Rules:
1. Use only premise IDs that appear in the problem.
2. If changing a claimed causal premise did not affect the answer, remove or reconsider that premise.
3. If changing an irrelevant premise affected the answer, ignore the distractor and revise toward the proof-relevant premises.
4. If contradiction is detected, answer unknown unless the problem clearly supports only one side.

Return valid JSON only:
{{
  "answer": "true | false | unknown",
  "causal_premises": ["premise ids necessary for the repaired answer"],
  "brief_explanation": "one short sentence"
}}
"""


def _safe_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
