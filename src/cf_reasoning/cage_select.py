from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .counterfactuals import generate_counterfactuals
from .schema import Example, Prediction


@dataclass(frozen=True)
class Candidate:
    answer: str
    causal_premises: tuple[str, ...]
    explanation: str
    raw: str


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    score: float
    diagnostics: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class CAGESelectConfig:
    n_candidates: int = 3
    max_counterfactuals: int = 4
    max_tokens: int = 1024


def cage_select_predict(
    example: Example,
    call_llm: Callable[[str, int, bool, dict | None], str],
    extract_label: Callable[[str], str],
    extract_premises: Callable[[str], tuple[str, ...]],
    config: CAGESelectConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or CAGESelectConfig()
    raw_rows: list[dict[str, str]] = []

    raw_candidates = call_llm(_candidate_prompt(example, config.n_candidates), config.max_tokens, True, None)
    raw_rows.append({"example_id": example.id, "method": "llm_cage_select_candidates", "raw_response": raw_candidates})
    candidates = _parse_candidates(raw_candidates)
    if not candidates:
        candidates = [Candidate(extract_label(raw_candidates), extract_premises(raw_candidates), "fallback parsed single output", raw_candidates)]

    scored = [
        _score_candidate(example, candidate, call_llm, extract_label, config)
        for candidate in candidates[: config.n_candidates]
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    best = scored[0]

    repair_raw = call_llm(_selection_repair_prompt(example, best, scored), config.max_tokens, True, None)
    raw_rows.append({"example_id": example.id, "method": "llm_cage_select", "raw_response": repair_raw})
    repaired_answer = extract_label(repair_raw)
    repaired_premises = extract_premises(repair_raw)
    if repaired_answer == "invalid":
        repaired_answer = best.candidate.answer
        repaired_premises = best.candidate.causal_premises

    trace = {
        "selected_score": best.score,
        "selected_initial_answer": best.candidate.answer,
        "selected_initial_premises": list(best.candidate.causal_premises),
        "diagnostics": list(best.diagnostics),
        "candidate_scores": [item.score for item in scored],
        "repair_raw": _safe_json(repair_raw),
    }
    return Prediction(example.id, "llm_cage_select", repaired_answer, repaired_premises, json.dumps(trace, sort_keys=True)), raw_rows


def _candidate_prompt(example: Example, n_candidates: int) -> str:
    return f"""You are solving a logical reasoning problem. Generate {n_candidates} diverse candidate answers.

Problem:
{example.text}

For each candidate, identify only the premise IDs that are causally necessary for the answer. Do not include irrelevant premises.

Return valid JSON only with this schema:
{{
  "candidates": [
    {{
      "answer": "true | false | unknown",
      "causal_premises": ["premise ids necessary for this answer"],
      "brief_explanation": "one short sentence"
    }}
  ]
}}
"""


def _score_candidate(
    example: Example,
    candidate: Candidate,
    call_llm: Callable[[str, int, bool, dict | None], str],
    extract_label: Callable[[str], str],
    config: CAGESelectConfig,
) -> ScoredCandidate:
    cfs = generate_counterfactuals([example], seed=23, max_per_example=config.max_counterfactuals)
    diagnostics: list[dict[str, str]] = []
    correct = 0
    total = 0
    claimed = set(candidate.causal_premises)
    for cf in cfs:
        raw = call_llm(_counterfactual_prompt(cf, candidate), min(config.max_tokens, 512), True, None)
        pred = extract_label(raw)
        expected_ok = _expected_ok(cf, candidate.answer, pred)
        failure = "ok" if expected_ok else _failure_type(cf, claimed, pred)
        diagnostics.append(
            {
                "cf_family": cf.cf_family,
                "intervention_type": cf.intervention_type,
                "changed_ids": ",".join(cf.changed_ids),
                "expected_relation": cf.expected_relation,
                "candidate_answer": candidate.answer,
                "counterfactual_answer": pred,
                "failure_type": failure,
            }
        )
        total += 1
        correct += int(expected_ok)
    base_valid = 1.0 if candidate.answer in {"true", "false", "unknown"} else 0.0
    premise_bonus = min(len(candidate.causal_premises), 6) / 6 if candidate.causal_premises else 0.0
    score = (correct / total if total else 0.0) * 0.75 + base_valid * 0.15 + premise_bonus * 0.10
    return ScoredCandidate(candidate, score, tuple(diagnostics))


def _counterfactual_prompt(cf, candidate: Candidate) -> str:
    return f"""You are evaluating a candidate logical answer under a counterfactual intervention.

Candidate original answer: {candidate.answer}
Candidate causal premises: {', '.join(candidate.causal_premises) if candidate.causal_premises else 'none'}
Candidate explanation: {candidate.explanation}
Intervention changed premise IDs: {', '.join(cf.changed_ids)}
Expected relation to original answer: {cf.expected_relation}

Counterfactual problem:
{cf.text}

Answer the counterfactual problem from scratch. Return valid JSON only:
{{
  "answer": "true | false | unknown",
  "brief_explanation": "one short sentence"
}}
"""


def _selection_repair_prompt(example: Example, selected: ScoredCandidate, all_scored: list[ScoredCandidate]) -> str:
    failed = [d for d in selected.diagnostics if d["failure_type"] != "ok"]
    failure_text = "\n".join(
        f"- {d['failure_type']}: {d['cf_family']} changed {d['changed_ids']}; expected {d['expected_relation']}; counterfactual answer {d['counterfactual_answer']}"
        for d in failed
    ) or "- The selected candidate passed the counterfactual checks. Keep it unless direct re-checking finds an error."
    score_text = ", ".join(f"{item.score:.2f}" for item in all_scored)
    return f"""You are revising the best candidate answer using counterfactual attribution evidence.

Problem:
{example.text}

Selected candidate:
- answer: {selected.candidate.answer}
- causal premises: {', '.join(selected.candidate.causal_premises) if selected.candidate.causal_premises else 'none'}
- explanation: {selected.candidate.explanation}
- counterfactual stability score: {selected.score:.2f}
All candidate scores: {score_text}

Failed checks for the selected candidate:
{failure_text}

Produce the final repaired answer. Use only premise IDs from the problem. Prefer a concise necessary support set. Return valid JSON only:
{{
  "answer": "true | false | unknown",
  "causal_premises": ["premise ids necessary for the final answer"],
  "brief_explanation": "one short sentence"
}}
"""


def _expected_ok(cf, original_answer: str, counterfactual_answer: str) -> bool:
    if counterfactual_answer == "invalid":
        return False
    if cf.conflict_label or cf.cf_family == "contradiction_injection":
        return counterfactual_answer in {"unknown", "ambiguous"}
    if cf.expected_relation == "changed":
        return counterfactual_answer != original_answer
    if cf.expected_relation == "preserved":
        return counterfactual_answer == original_answer
    return False


def _failure_type(cf, claimed: set[str], pred: str) -> str:
    changed = set(cf.changed_ids)
    if cf.conflict_label or cf.cf_family == "contradiction_injection":
        return "missed_contradiction"
    if cf.expected_relation == "changed":
        return "insensitive_to_claimed_causal_change" if changed & claimed else "missed_proof_break"
    if cf.expected_relation == "preserved":
        return "distractor_sensitive" if not (changed & claimed) else "unstable_claimed_support"
    return "counterfactual_failure"


def _parse_candidates(text: str) -> list[Candidate]:
    data = _json_from_text(text)
    items = data.get("candidates", []) if isinstance(data, dict) else []
    candidates: list[Candidate] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            answer = str(item.get("answer", "invalid")).lower().strip()
            premises = item.get("causal_premises", [])
            if not isinstance(premises, list):
                premises = []
            explanation = str(item.get("brief_explanation", ""))
            candidates.append(Candidate(answer, tuple(str(p) for p in premises), explanation, json.dumps(item, sort_keys=True)))
    return candidates


def _json_from_text(text: str):
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _safe_json(text: str):
    data = _json_from_text(text)
    return data if data is not None else text
