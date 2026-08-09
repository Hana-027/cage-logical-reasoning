from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Iterable

from .datasets import FolioExample
from .schema import CounterfactualExample, Example, Prediction
from .logiclm_adapter import LogicLMConfig, folio_logiclm_predict, logiclm_predict
from .method_cage import folio_cage_wrap, folio_cage_wrap_gated, structured_cage_wrap, structured_cage_wrap_gated
from .prompts import direct_prompt
from .symbcot_adapter import SymbCoTConfig, folio_symbcot_predict, symbcot_predict

SUPPORTED_BASELINES = ("logiclm", "symbcot", "vericot", "direct_cage", "logiclm_cage", "symbcot_cage", "direct_cage_gated", "logiclm_cage_gated", "symbcot_cage_gated")

BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["true", "false", "unknown"]},
        "causal_premises": {"type": "array", "items": {"type": "string"}},
        "brief_explanation": {"type": "string"},
    },
    "required": ["answer", "causal_premises", "brief_explanation"],
    "additionalProperties": False,
}
FOLIO_BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["true", "false", "unknown"]},
        "brief_explanation": {"type": "string"},
    },
    "required": ["answer", "brief_explanation"],
    "additionalProperties": False,
}
VERICOT_SCHEMA = {
    "type": "object",
    "properties": {
        "verified": {"type": "boolean"},
        "answer": {"type": "string", "enum": ["true", "false", "unknown"]},
        "causal_premises": {"type": "array", "items": {"type": "string"}},
        "error_type": {
            "type": "string",
            "enum": ["none", "invalid_derivation", "missed_contradiction", "unsupported_answer", "uncertain"],
        },
        "brief_explanation": {"type": "string"},
    },
    "required": ["verified", "answer", "causal_premises", "error_type", "brief_explanation"],
    "additionalProperties": False,
}
FOLIO_VERICOT_SCHEMA = {
    "type": "object",
    "properties": {
        "verified": {"type": "boolean"},
        "answer": {"type": "string", "enum": ["true", "false", "unknown"]},
        "error_type": {
            "type": "string",
            "enum": ["none", "invalid_derivation", "missed_contradiction", "unsupported_answer", "uncertain"],
        },
        "brief_explanation": {"type": "string"},
    },
    "required": ["verified", "answer", "error_type", "brief_explanation"],
    "additionalProperties": False,
}

CallLLM = Callable[[str, int, bool, dict | None], str]
ExtractLabel = Callable[[str], str]
ExtractPremises = Callable[[str], tuple[str, ...]]


BaseCache = dict[tuple[str, str], tuple[Prediction, list[dict[str, str]]]]


@dataclass(frozen=True)
class LLMBaselineConfig:
    max_tokens: int = 512


def normalize_baseline_methods(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_methods = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not raw_methods or raw_methods == ("none",):
        return ()
    methods = SUPPORTED_BASELINES if raw_methods == ("all",) else raw_methods
    invalid = [method for method in methods if method not in SUPPORTED_BASELINES]
    if invalid:
        raise ValueError(f"Unsupported LLM baseline method(s): {', '.join(invalid)}. Use logiclm, symbcot, vericot, direct_cage, logiclm_cage, symbcot_cage, direct_cage_gated, logiclm_cage_gated, symbcot_cage_gated, or all.")
    return tuple(dict.fromkeys(methods))


def llm_baseline_predictions(
    examples: list[Example] | list[CounterfactualExample],
    methods: Iterable[str],
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    extract_premises: ExtractPremises,
    config: LLMBaselineConfig | None = None,
    base_predictions: BaseCache | None = None,
) -> tuple[list[Prediction], list[dict[str, str]]]:
    config = config or LLMBaselineConfig()
    cache: BaseCache = dict(base_predictions or {})
    predictions: list[Prediction] = []
    raw_rows: list[dict[str, str]] = []
    for ex in examples:
        for method in methods:
            pred, raw = _predict_general_baseline(ex, method, call_llm, extract_label, extract_premises, config, cache)
            predictions.append(pred)
            raw_rows.extend(raw)
            cache.setdefault((ex.id, pred.method), (pred, raw))
    return predictions, raw_rows


def folio_llm_baseline_rows(
    examples: list[FolioExample],
    methods: Iterable[str],
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: LLMBaselineConfig | None = None,
    base_predictions: BaseCache | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    config = config or LLMBaselineConfig()
    cache: BaseCache = dict(base_predictions or {})
    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, str]] = []
    for ex in examples:
        for method in methods:
            pred, raw = _predict_folio_baseline(ex, method, call_llm, extract_label, config, cache)
            rows.append({"split": ex.split, "example_id": ex.id, "method": pred.method, "gold": ex.label, "pred": pred.label, "accuracy": int(pred.label == ex.label)})
            raw_rows.extend(raw)
            cache.setdefault((ex.id, pred.method), (pred, raw))
    return rows, raw_rows


def _cached_or_predict(
    cache: BaseCache,
    example_id: str,
    method: str,
    predict: Callable[[], tuple[Prediction, list[dict[str, str]]]],
) -> tuple[Prediction, list[dict[str, str]]]:
    cached = cache.get((example_id, method))
    if cached is not None:
        return cached[0], []
    pred, raw = predict()
    cache[(example_id, method)] = (pred, raw)
    return pred, raw


def _predict_general_baseline(
    ex: Example | CounterfactualExample,
    method: str,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    extract_premises: ExtractPremises,
    config: LLMBaselineConfig,
    cache: BaseCache,
) -> tuple[Prediction, list[dict[str, str]]]:
    if method == "logiclm":
        return _cached_or_predict(
            cache,
            ex.id,
            "llm_logiclm",
            lambda: logiclm_predict(ex, call_llm, extract_label, extract_premises, LogicLMConfig(max_tokens=config.max_tokens)),
        )
    if method == "logiclm_cage":
        return structured_cage_wrap(
            ex,
            "llm_logiclm",
            "llm_logiclm_cage",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_logiclm",
                lambda: logiclm_predict(item, call_llm, extract_label, extract_premises, LogicLMConfig(max_tokens=config.max_tokens)),
            ),
            call_llm,
            extract_label,
            extract_premises,
            config.max_tokens,
        )
    if method == "logiclm_cage_gated":
        return structured_cage_wrap_gated(
            ex,
            "llm_logiclm",
            "llm_logiclm_cage_gated",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_logiclm",
                lambda: logiclm_predict(item, call_llm, extract_label, extract_premises, LogicLMConfig(max_tokens=config.max_tokens)),
            ),
            call_llm,
            extract_label,
            extract_premises,
            config.max_tokens,
        )
    if method == "direct_cage":
        return structured_cage_wrap(
            ex,
            "llm_direct",
            "llm_direct_cage",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_direct",
                lambda: _direct_predict(item, call_llm, extract_label, extract_premises, config),
            ),
            call_llm,
            extract_label,
            extract_premises,
            config.max_tokens,
        )
    if method == "direct_cage_gated":
        return structured_cage_wrap_gated(
            ex,
            "llm_direct",
            "llm_direct_cage_gated",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_direct",
                lambda: _direct_predict(item, call_llm, extract_label, extract_premises, config),
            ),
            call_llm,
            extract_label,
            extract_premises,
            config.max_tokens,
        )
    if method == "symbcot":
        return _cached_or_predict(
            cache,
            ex.id,
            "llm_symbcot",
            lambda: symbcot_predict(ex, call_llm, extract_label, extract_premises, SymbCoTConfig(max_tokens=config.max_tokens)),
        )
    if method == "symbcot_cage":
        return structured_cage_wrap(
            ex,
            "llm_symbcot",
            "llm_symbcot_cage",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_symbcot",
                lambda: symbcot_predict(item, call_llm, extract_label, extract_premises, SymbCoTConfig(max_tokens=config.max_tokens)),
            ),
            call_llm,
            extract_label,
            extract_premises,
            config.max_tokens,
        )
    if method == "symbcot_cage_gated":
        return structured_cage_wrap_gated(
            ex,
            "llm_symbcot",
            "llm_symbcot_cage_gated",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_symbcot",
                lambda: symbcot_predict(item, call_llm, extract_label, extract_premises, SymbCoTConfig(max_tokens=config.max_tokens)),
            ),
            call_llm,
            extract_label,
            extract_premises,
            config.max_tokens,
        )
    if method == "vericot":
        return _predict_general_vericot(ex, call_llm, extract_label, extract_premises, config)
    raise ValueError(f"Unsupported LLM baseline method: {method}")


def _predict_folio_baseline(
    ex: FolioExample,
    method: str,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: LLMBaselineConfig,
    cache: BaseCache,
) -> tuple[Prediction, list[dict[str, str]]]:
    if method == "logiclm":
        pred, raw = _cached_or_predict(
            cache,
            ex.id,
            "llm_logiclm",
            lambda: folio_logiclm_predict(ex, call_llm, extract_label, LogicLMConfig(max_tokens=config.max_tokens)),
        )
        return pred, raw
    if method == "logiclm_cage":
        return folio_cage_wrap(
            ex,
            "llm_logiclm",
            "llm_logiclm_cage",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_logiclm",
                lambda: folio_logiclm_predict(item, call_llm, extract_label, LogicLMConfig(max_tokens=config.max_tokens)),
            ),
            call_llm,
            extract_label,
            config.max_tokens,
        )
    if method == "logiclm_cage_gated":
        return folio_cage_wrap_gated(
            ex,
            "llm_logiclm",
            "llm_logiclm_cage_gated",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_logiclm",
                lambda: folio_logiclm_predict(item, call_llm, extract_label, LogicLMConfig(max_tokens=config.max_tokens)),
            ),
            call_llm,
            extract_label,
            config.max_tokens,
        )
    if method == "direct_cage":
        return folio_cage_wrap(
            ex,
            "llm_direct",
            "llm_direct_cage",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_direct",
                lambda: _folio_direct_predict(item, call_llm, extract_label, config),
            ),
            call_llm,
            extract_label,
            config.max_tokens,
        )
    if method == "direct_cage_gated":
        return folio_cage_wrap_gated(
            ex,
            "llm_direct",
            "llm_direct_cage_gated",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_direct",
                lambda: _folio_direct_predict(item, call_llm, extract_label, config),
            ),
            call_llm,
            extract_label,
            config.max_tokens,
        )
    if method == "symbcot":
        pred, raw = _cached_or_predict(
            cache,
            ex.id,
            "llm_symbcot",
            lambda: folio_symbcot_predict(ex, call_llm, extract_label, SymbCoTConfig(max_tokens=config.max_tokens)),
        )
        return pred, raw
    if method == "symbcot_cage":
        return folio_cage_wrap(
            ex,
            "llm_symbcot",
            "llm_symbcot_cage",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_symbcot",
                lambda: folio_symbcot_predict(item, call_llm, extract_label, SymbCoTConfig(max_tokens=config.max_tokens)),
            ),
            call_llm,
            extract_label,
            config.max_tokens,
        )
    if method == "symbcot_cage_gated":
        return folio_cage_wrap_gated(
            ex,
            "llm_symbcot",
            "llm_symbcot_cage_gated",
            lambda item: _cached_or_predict(
                cache,
                item.id,
                "llm_symbcot",
                lambda: folio_symbcot_predict(item, call_llm, extract_label, SymbCoTConfig(max_tokens=config.max_tokens)),
            ),
            call_llm,
            extract_label,
            config.max_tokens,
        )
    if method == "vericot":
        return _predict_folio_vericot(ex, call_llm, extract_label, config)
    raise ValueError(f"Unsupported LLM baseline method: {method}")


def _direct_predict(
    ex: Example | CounterfactualExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    extract_premises: ExtractPremises,
    config: LLMBaselineConfig,
) -> tuple[Prediction, list[dict[str, str]]]:
    raw = call_llm(direct_prompt(ex), config.max_tokens, False, None)
    return Prediction(ex.id, "llm_direct", extract_label(raw), extract_premises(raw), raw), [
        {"example_id": ex.id, "method": "llm_direct", "raw_response": raw}
    ]


def _folio_direct_predict(
    ex: FolioExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: LLMBaselineConfig,
) -> tuple[Prediction, list[dict[str, str]]]:
    prompt = f"""You are solving a FOLIO logical reasoning problem.

Premises:
{ex.text}

Formal representation:
{chr(10).join(f'FOL{i + 1}: {premise}' for i, premise in enumerate(ex.premises_fol))}
Conclusion-FOL: {ex.conclusion_fol}

Answer with exactly one of: true, false, unknown.
Final answer:"""
    raw = call_llm(prompt, config.max_tokens, False, None)
    return Prediction(ex.id, "llm_direct", extract_label(raw), (), raw), [
        {"example_id": ex.id, "method": "llm_direct", "raw_response": raw}
    ]


def _predict_general_vericot(
    ex: Example | CounterfactualExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    extract_premises: ExtractPremises,
    config: LLMBaselineConfig,
) -> tuple[Prediction, list[dict[str, str]]]:
    draft_raw = call_llm(_vericot_draft_prompt(ex), config.max_tokens, True, BASELINE_SCHEMA)
    draft_label = extract_label(draft_raw)
    draft_premises = extract_premises(draft_raw)
    verify_raw = call_llm(_vericot_verify_prompt(ex, draft_raw, draft_label, draft_premises), config.max_tokens, True, VERICOT_SCHEMA)
    final_label = extract_label(verify_raw)
    if final_label == "invalid" and _verification_keeps_draft(verify_raw):
        final_label = draft_label
    final_premises = extract_premises(verify_raw)
    if not final_premises and final_label == draft_label:
        final_premises = draft_premises
    trace = _trace(draft_raw, draft_label, draft_premises, verify_raw, final_label, final_premises)
    return Prediction(ex.id, "llm_vericot", final_label, final_premises, trace), [
        _raw_row(ex.id, "llm_vericot_draft", draft_raw),
        _raw_row(ex.id, "llm_vericot_verify", verify_raw),
    ]


def _predict_folio_vericot(
    ex: FolioExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: LLMBaselineConfig,
) -> tuple[Prediction, list[dict[str, str]]]:
    draft_raw = call_llm(_folio_vericot_draft_prompt(ex), config.max_tokens, True, FOLIO_BASELINE_SCHEMA)
    draft_label = extract_label(draft_raw)
    verify_raw = call_llm(_folio_vericot_verify_prompt(ex, draft_raw, draft_label), config.max_tokens, True, FOLIO_VERICOT_SCHEMA)
    final_label = extract_label(verify_raw)
    if final_label == "invalid" and _verification_keeps_draft(verify_raw):
        final_label = draft_label
    trace = _trace(draft_raw, draft_label, (), verify_raw, final_label, ())
    return Prediction(ex.id, "llm_vericot", final_label, (), trace), [
        _raw_row(ex.id, "llm_vericot_draft", draft_raw),
        _raw_row(ex.id, "llm_vericot_verify", verify_raw),
    ]


def _verification_keeps_draft(raw: str) -> bool:
    data = _json_from_text(raw)
    if not isinstance(data, dict):
        return False
    if data.get("is_valid") is True or data.get("verified") is True:
        return not any(key in data and data[key] for key in ["answer", "final_answer", "final_label", "corrected_answer", "revised_answer", "revision"])
    return False


def _logiclm_prompt(example: Example | CounterfactualExample) -> str:
    return f"""You are implementing a LogicLM-style logical reasoning baseline.

Translate the natural-language problem into a compact symbolic formulation, reason over that formulation as if using a deterministic solver, and return the final label.

Problem:
{example.text}

Return only JSON matching the requested schema. Use causal_premises for the premise/rule IDs that support the answer."""


def _symbcot_prompt(example: Example | CounterfactualExample) -> str:
    return f"""You are implementing a Symbolic Chain-of-Thought logical reasoning baseline.

Create a concise public symbolic proof sketch using the premise and rule IDs, check whether the query or its negation follows, and return the final label.

Problem:
{example.text}

Return only JSON matching the requested schema. Keep brief_explanation short and proof-checkable."""


def _vericot_draft_prompt(example: Example | CounterfactualExample) -> str:
    return f"""You are drafting an answer for a VeriCoT-style verification baseline.

Produce a concise proof sketch grounded in the listed facts and rules. The draft will be checked by a verifier, so include the premise/rule IDs that support your answer.

Problem:
{example.text}

Return only JSON matching the requested schema."""


def _vericot_verify_prompt(
    example: Example | CounterfactualExample,
    draft_raw: str,
    draft_label: str,
    draft_premises: tuple[str, ...],
) -> str:
    return f"""You are the verifier in a VeriCoT-style baseline.

Check whether the draft answer is logically supported by the problem. If the draft is unsupported, misses a contradiction, or reaches the wrong label, revise the answer.

Problem:
{example.text}

Draft label: {draft_label}
Draft support IDs: {', '.join(draft_premises) if draft_premises else '(none)'}
Draft response:
{draft_raw}

Return only JSON matching the requested schema."""


def _folio_logiclm_prompt(example: FolioExample) -> str:
    return f"""You are implementing a LogicLM-style baseline for FOLIO.

Use the formal representation as the symbolic formulation, reason over it as if using a deterministic first-order solver, and return whether the conclusion is true, false, or unknown.

{_folio_problem(example)}

Return only JSON matching the requested schema."""


def _folio_symbcot_prompt(example: FolioExample) -> str:
    return f"""You are implementing a Symbolic Chain-of-Thought baseline for FOLIO.

Use the natural-language premises and formal representation to create a concise public symbolic proof sketch. Then return whether the conclusion is true, false, or unknown.

{_folio_problem(example)}

Return only JSON matching the requested schema."""


def _folio_vericot_draft_prompt(example: FolioExample) -> str:
    return f"""You are drafting an answer for a VeriCoT-style FOLIO baseline.

Draft a concise, checkable argument grounded in the premises and formal representation, then return the final label.

{_folio_problem(example)}

Return only JSON matching the requested schema."""


def _folio_vericot_verify_prompt(example: FolioExample, draft_raw: str, draft_label: str) -> str:
    return f"""You are the verifier in a VeriCoT-style FOLIO baseline.

Check whether the draft label is supported by the premises and formal representation. If the draft is unsupported, misses a contradiction, or should be unknown, revise the answer.

{_folio_problem(example)}

Draft label: {draft_label}
Draft response:
{draft_raw}

Return only JSON matching the requested schema."""


def _folio_problem(example: FolioExample) -> str:
    fol_lines = [f"FOL{i + 1}: {premise}" for i, premise in enumerate(example.premises_fol)]
    fol_lines.append(f"Conclusion-FOL: {example.conclusion_fol}")
    return f"""Natural-language problem:
{example.text}

Formal representation:
{chr(10).join(fol_lines)}"""


def _raw_row(example_id: str, method: str, raw: str) -> dict[str, str]:
    return {"example_id": example_id, "method": method, "raw_response": raw}


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


def _trace(
    draft_raw: str,
    draft_label: str,
    draft_premises: tuple[str, ...],
    verify_raw: str,
    final_label: str,
    final_premises: tuple[str, ...],
) -> str:
    return json.dumps(
        {
            "draft_raw": draft_raw,
            "draft_answer": draft_label,
            "draft_premises": list(draft_premises),
            "verification_raw": verify_raw,
            "final_answer": final_label,
            "final_premises": list(final_premises),
        },
        sort_keys=True,
    )
