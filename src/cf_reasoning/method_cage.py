from __future__ import annotations

import json
from collections import Counter
from typing import Callable

from .cage import CAGEConfig, _classify_failure, _diagnose, _repair_prompt
from .folio_cage import FolioCAGEConfig, _extract_premise_ids, _folio_probe_diagnostics, _folio_repair_prompt
from .schema import CounterfactualExample, Example, Prediction
from .datasets import FolioExample

CallLLM = Callable[[str, int, bool, dict | None], str]
ExtractLabel = Callable[[str], str]
ExtractPremises = Callable[[str], tuple[str, ...]]
BasePredict = Callable[[Example | CounterfactualExample | FolioExample], tuple[Prediction, list[dict[str, str]]]]
VALID_LABELS = {"true", "false", "unknown"}


def structured_cage_wrap(
    example: Example | CounterfactualExample,
    base_method: str,
    caged_method: str,
    base_predict: BasePredict,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    extract_premises: ExtractPremises,
    max_tokens: int,
    max_counterfactuals: int = 3,
) -> tuple[Prediction, list[dict[str, str]]]:
    base_pred, raw_rows = base_predict(example)
    initial_label = base_pred.label
    initial_premises = base_pred.premise_ids
    diagnostics = []
    if isinstance(example, Example):
        diagnostics = _diagnose(
            example,
            initial_label,
            initial_premises,
            call_llm,
            extract_label,
            CAGEConfig(max_counterfactuals=max_counterfactuals, max_tokens=max_tokens),
        )
    repair_triggered = initial_label not in VALID_LABELS
    repair_raw = ""
    final_label = initial_label
    final_premises = initial_premises
    if repair_triggered:
        repair_raw = call_llm(
            _repair_prompt(example, base_pred.raw_response, initial_label, initial_premises, diagnostics),
            max_tokens,
            True,
            None,
        )
        repaired_label = extract_label(repair_raw)
        if repaired_label in VALID_LABELS:
            final_label = repaired_label
            repaired_premises = extract_premises(repair_raw)
            if repaired_premises:
                final_premises = repaired_premises
        raw_rows.append(_raw_row(example.id, caged_method, repair_raw))
    trace = {
        "base_method": base_method,
        "base_raw": _safe_json(base_pred.raw_response),
        "initial_answer": initial_label,
        "initial_premises": list(initial_premises),
        "diagnostics": diagnostics,
        "repair_triggered": repair_triggered,
        "repair_reason": "invalid_initial_label" if repair_triggered else "valid_initial_label",
        "repair_raw": _safe_json(repair_raw) if repair_raw else None,
        "final_answer": final_label,
    }
    return Prediction(example.id, caged_method, final_label, final_premises, json.dumps(trace, sort_keys=True)), raw_rows


def structured_cage_wrap_gated(
    example: Example | CounterfactualExample,
    base_method: str,
    caged_method: str,
    base_predict: BasePredict,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    extract_premises: ExtractPremises,
    max_tokens: int,
    max_counterfactuals: int = 3,
    min_failed: int = 2,
) -> tuple[Prediction, list[dict[str, str]]]:
    base_pred, raw_rows = base_predict(example)
    initial_label = base_pred.label
    initial_premises = base_pred.premise_ids
    diagnostics = []
    if isinstance(example, Example):
        diagnostics = _diagnose(
            example,
            initial_label,
            initial_premises,
            call_llm,
            extract_label,
            CAGEConfig(max_counterfactuals=max_counterfactuals, max_tokens=max_tokens),
        )
    gated_label, gate_reason = _structured_gated_label(initial_label, diagnostics, min_failed)
    repair_triggered = initial_label not in VALID_LABELS or gated_label is not None
    repair_raw = ""
    final_label = initial_label
    final_premises = initial_premises
    repair_accepted = False
    if repair_triggered:
        repair_raw = call_llm(
            _repair_prompt(example, base_pred.raw_response, initial_label, initial_premises, diagnostics),
            max_tokens,
            True,
            None,
        )
        repaired_label = extract_label(repair_raw)
        if repaired_label in VALID_LABELS and (gated_label is None or repaired_label == gated_label):
            final_label = repaired_label
            repair_accepted = True
            repaired_premises = extract_premises(repair_raw)
            if repaired_premises:
                final_premises = repaired_premises
        raw_rows.append(_raw_row(example.id, caged_method, repair_raw))
    trace = {
        "base_method": base_method,
        "base_raw": _safe_json(base_pred.raw_response),
        "initial_answer": initial_label,
        "initial_premises": list(initial_premises),
        "diagnostics": diagnostics,
        "repair_triggered": repair_triggered,
        "repair_accepted": repair_accepted,
        "repair_reason": gate_reason,
        "gated_target_answer": gated_label,
        "repair_raw": _safe_json(repair_raw) if repair_raw else None,
        "final_answer": final_label,
    }
    return Prediction(example.id, caged_method, final_label, final_premises, json.dumps(trace, sort_keys=True)), raw_rows


def folio_cage_wrap(
    example,
    base_method: str,
    caged_method: str,
    base_predict: BasePredict,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    max_tokens: int,
) -> tuple[Prediction, list[dict[str, str]]]:
    base_pred, raw_rows = base_predict(example)
    initial_label = base_pred.label
    diagnostics = _folio_probe_diagnostics(
        example,
        base_pred.raw_response,
        initial_label,
        call_llm,
        extract_label,
        FolioCAGEConfig(max_tokens=max_tokens),
    )
    repair_triggered = initial_label not in VALID_LABELS
    repair_raw = ""
    final_label = initial_label
    final_premises = base_pred.premise_ids
    if repair_triggered:
        repair_raw = call_llm(
            _folio_repair_prompt(example, base_pred.raw_response, initial_label, diagnostics),
            max_tokens,
            True,
            None,
        )
        repaired_label = extract_label(repair_raw)
        if repaired_label in VALID_LABELS:
            final_label = repaired_label
            repaired_premises = _extract_premise_ids(repair_raw)
            if repaired_premises:
                final_premises = repaired_premises
        raw_rows.append(_raw_row(example.id, caged_method, repair_raw))
    trace = {
        "base_method": base_method,
        "base_raw": _safe_json(base_pred.raw_response),
        "initial_answer": initial_label,
        "initial_premises": list(base_pred.premise_ids),
        "diagnostics": diagnostics,
        "repair_triggered": repair_triggered,
        "repair_reason": "invalid_initial_label" if repair_triggered else "valid_initial_label",
        "repair_raw": _safe_json(repair_raw) if repair_raw else None,
        "final_answer": final_label,
    }
    return Prediction(example.id, caged_method, final_label, final_premises, json.dumps(trace, sort_keys=True)), raw_rows


def folio_cage_wrap_gated(
    example,
    base_method: str,
    caged_method: str,
    base_predict: BasePredict,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    max_tokens: int,
    min_failed: int = 2,
) -> tuple[Prediction, list[dict[str, str]]]:
    base_pred, raw_rows = base_predict(example)
    initial_label = base_pred.label
    diagnostics = _folio_probe_diagnostics(
        example,
        base_pred.raw_response,
        initial_label,
        call_llm,
        extract_label,
        FolioCAGEConfig(max_tokens=max_tokens),
    )
    gated_label, gate_reason = _folio_gated_label(initial_label, diagnostics, min_failed)
    repair_triggered = initial_label not in VALID_LABELS or gated_label is not None
    repair_raw = ""
    final_label = initial_label
    final_premises = base_pred.premise_ids
    repair_accepted = False
    if repair_triggered:
        repair_raw = call_llm(
            _folio_repair_prompt(example, base_pred.raw_response, initial_label, diagnostics),
            max_tokens,
            True,
            None,
        )
        repaired_label = extract_label(repair_raw)
        if repaired_label in VALID_LABELS and (gated_label is None or repaired_label == gated_label):
            final_label = repaired_label
            repair_accepted = True
            repaired_premises = _extract_premise_ids(repair_raw)
            if repaired_premises:
                final_premises = repaired_premises
        raw_rows.append(_raw_row(example.id, caged_method, repair_raw))
    trace = {
        "base_method": base_method,
        "base_raw": _safe_json(base_pred.raw_response),
        "initial_answer": initial_label,
        "initial_premises": list(base_pred.premise_ids),
        "diagnostics": diagnostics,
        "repair_triggered": repair_triggered,
        "repair_accepted": repair_accepted,
        "repair_reason": gate_reason,
        "gated_target_answer": gated_label,
        "repair_raw": _safe_json(repair_raw) if repair_raw else None,
        "final_answer": final_label,
    }
    return Prediction(example.id, caged_method, final_label, final_premises, json.dumps(trace, sort_keys=True)), raw_rows


def _structured_gated_label(initial_label: str, diagnostics: list[dict[str, str]], min_failed: int) -> tuple[str | None, str]:
    if initial_label not in VALID_LABELS:
        return None, "invalid_initial_label"
    failed = [d for d in diagnostics if d.get("failure_type") not in {None, "ok"}]
    if len(failed) < min_failed:
        return None, "insufficient_failed_diagnostics"
    answers = [d.get("counterfactual_answer") for d in failed]
    return _majority_alternative(initial_label, answers, "strong_structured_diagnostic_majority")


def _folio_gated_label(initial_label: str, diagnostics: list[dict[str, str]], min_failed: int) -> tuple[str | None, str]:
    if initial_label not in VALID_LABELS:
        return None, "invalid_initial_label"
    failed = [d for d in diagnostics if d.get("status") in {"disagree", "conflict"}]
    if len(failed) < min_failed:
        return None, "insufficient_failed_diagnostics"
    answers = [d.get("probe_answer") for d in failed]
    return _majority_alternative(initial_label, answers, "strong_folio_diagnostic_majority")


def _majority_alternative(initial_label: str, answers: list[str | None], success_reason: str) -> tuple[str | None, str]:
    counts = Counter(answer for answer in answers if answer in VALID_LABELS and answer != initial_label)
    if not counts:
        return None, "no_valid_alternative_majority"
    label, count = counts.most_common(1)[0]
    if count * 2 <= len(answers):
        return None, "no_strict_alternative_majority"
    return label, success_reason


def _safe_json(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _raw_row(example_id: str, method: str, raw: str) -> dict[str, str]:
    return {"example_id": example_id, "method": method, "raw_response": raw}
