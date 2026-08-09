from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Callable

from .counterfactuals import generate_counterfactuals
from .datasets import FolioExample
from .schema import CounterfactualExample, Example, Prediction

CallLLM = Callable[[str, int, bool, dict | None], str]
ExtractLabel = Callable[[str], str]
ExtractPremises = Callable[[str], tuple[str, ...]]

ROOT = Path(__file__).resolve().parents[2]
SYMBCOT_ROOT = ROOT / "SymbCoT"
DATASET_ALIASES = {
    "proofwriter": "ProofWriter",
    "prontoqa": "ProntoQA",
    "folio": "FOLIO",
    "synthetic": "ProofWriter",
    "": "ProofWriter",
}


@dataclass(frozen=True)
class SymbCoTConfig:
    max_tokens: int = 512
    prompt_root: Path = SYMBCOT_ROOT / "prompts"
    max_counterfactuals: int = 3


def symbcot_predict(
    example: Example | CounterfactualExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    extract_premises: ExtractPremises,
    config: SymbCoTConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or SymbCoTConfig()
    dataset = _dataset_name(example.split)
    context, question = _example_problem_parts(example)
    trace, raw_rows = _run_symbcot_pipeline(example.id, dataset, context, question, call_llm, extract_label, config)
    label = trace["final_answer"]
    premises = extract_premises(trace["solver_raw"])
    return Prediction(example.id, "llm_symbcot", label, premises, json.dumps(trace, sort_keys=True)), raw_rows


def symbcot_cage_predict(
    example: Example | CounterfactualExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    extract_premises: ExtractPremises,
    config: SymbCoTConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or SymbCoTConfig()
    base_pred, raw_rows = symbcot_predict(example, call_llm, extract_label, extract_premises, config)
    initial_raw = base_pred.raw_response
    initial_label = base_pred.label
    initial_premises = base_pred.premise_ids
    diagnostics = _symbcot_counterfactual_diagnostics(example, initial_label, initial_premises, call_llm, extract_label, config)
    should_repair = _should_repair_symbcot_cage(initial_label, diagnostics)
    if should_repair:
        repair_raw = call_llm(_symbcot_cage_repair_prompt(example, initial_raw, initial_label, initial_premises, diagnostics), config.max_tokens, True, None)
        repaired_label = extract_label(repair_raw)
        repaired_premises = extract_premises(repair_raw)
        raw_rows.append(_raw_row(example.id, "llm_symbcot_cage", repair_raw))
        final_label = repaired_label if repaired_label in {"true", "false", "unknown"} else initial_label
        final_premises = repaired_premises if repaired_label in {"true", "false", "unknown"} else initial_premises
    else:
        repair_raw = ""
        final_label = initial_label
        final_premises = initial_premises
    trace = {
        "symbcot_trace": _safe_json(initial_raw),
        "initial_answer": initial_label,
        "initial_premises": list(initial_premises),
        "diagnostics": diagnostics,
        "repair_raw": _safe_json(repair_raw) if repair_raw else None,
        "final_answer": final_label,
    }
    return Prediction(example.id, "llm_symbcot_cage", final_label, final_premises, json.dumps(trace, sort_keys=True)), raw_rows


def folio_symbcot_predict(
    example: FolioExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: SymbCoTConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or SymbCoTConfig()
    context = " ".join(example.premises)
    question = _folio_question(example)
    trace, raw_rows = _run_symbcot_pipeline(example.id, "FOLIO", context, question, call_llm, extract_label, config)
    return Prediction(example.id, "llm_symbcot", trace["final_answer"], (), json.dumps(trace, sort_keys=True)), raw_rows


def folio_symbcot_cage_predict(
    example: FolioExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: SymbCoTConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or SymbCoTConfig()
    base_pred, raw_rows = folio_symbcot_predict(example, call_llm, extract_label, config)
    diagnostics = _folio_transfer_diagnostics(example, base_pred.raw_response, base_pred.label, call_llm, extract_label, config)
    failed = [d for d in diagnostics if d["status"] not in {"ok", "invalid"}]
    if failed:
        repair_raw = call_llm(_folio_symbcot_cage_repair_prompt(example, base_pred.raw_response, base_pred.label, diagnostics), config.max_tokens, True, None)
        repaired_label = extract_label(repair_raw)
        raw_rows.append(_raw_row(example.id, "llm_symbcot_cage", repair_raw))
        final_label = repaired_label if repaired_label in {"true", "false", "unknown"} else base_pred.label
    else:
        repair_raw = ""
        final_label = base_pred.label
    trace = {
        "symbcot_trace": _safe_json(base_pred.raw_response),
        "initial_answer": base_pred.label,
        "diagnostics": diagnostics,
        "repair_raw": _safe_json(repair_raw) if repair_raw else None,
        "final_answer": final_label,
    }
    return Prediction(example.id, "llm_symbcot_cage", final_label, (), json.dumps(trace, sort_keys=True)), raw_rows


def _run_symbcot_pipeline(
    example_id: str,
    dataset: str,
    context: str,
    question: str,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: SymbCoTConfig,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    translation_prompt = _template(config.prompt_root, dataset, "translation.txt").replace("[[CONTEXT]]", context).replace("[[QUESTION]]", question)
    translation_raw = call_llm(translation_prompt, config.max_tokens, False, None)

    plan_prompt = _template(config.prompt_root, dataset, "plan_generation.txt").replace("[[CONTEXT]]", translation_raw)
    plan_raw = call_llm(plan_prompt, config.max_tokens, False, None)

    solver_prompt = _template(config.prompt_root, dataset, "solver.txt").replace("[[CONTEXT]]", translation_raw).replace("[[PLAN]]", plan_raw)
    solver_raw = call_llm(solver_prompt, config.max_tokens, False, None)
    final_answer = _extract_final_answer(solver_raw, extract_label)

    trace = {
        "dataset": dataset,
        "translation_raw": translation_raw,
        "plan_raw": plan_raw,
        "solver_raw": solver_raw,
        "final_answer": final_answer,
    }
    raw_rows = [
        _raw_row(example_id, "llm_symbcot_translation", translation_raw),
        _raw_row(example_id, "llm_symbcot_plan", plan_raw),
        _raw_row(example_id, "llm_symbcot", solver_raw),
    ]
    return trace, raw_rows


def _template(prompt_root: Path, dataset: str, filename: str) -> str:
    path = prompt_root / dataset / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    fallback = SYMBCOT_ROOT / "prompts" / dataset / filename
    if fallback.exists():
        return fallback.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Missing SymbCoT prompt template: {path}")


def _dataset_name(split: str) -> str:
    normalized = (split or "").lower()
    for key, value in DATASET_ALIASES.items():
        if key and key in normalized:
            return value
    return DATASET_ALIASES.get(normalized, "ProofWriter")


def _example_problem_parts(example: Example | CounterfactualExample) -> tuple[str, str]:
    lines = [line.strip() for line in example.text.splitlines() if line.strip()]
    query_lines = [line for line in lines if line.lower().startswith("query:")]
    context_lines = [line for line in lines if not line.lower().startswith("query:")]
    query = query_lines[-1].removeprefix("Query:").strip() if query_lines else example.query.to_text().strip()
    return "\n".join(context_lines), query


def _folio_question(example: FolioExample) -> str:
    return f"Based on the above information, is the following statement true, false, or uncertain? {example.conclusion}"


def _extract_final_answer(text: str, extract_label: ExtractLabel) -> str:
    bracket = re.search(r"Final answer:\s*\{\s*([^}]+?)\s*\}", text, re.IGNORECASE)
    if bracket:
        label = extract_label(bracket.group(1))
        if label in {"true", "false", "unknown"}:
            return label
    generic = re.findall(r"\{\s*(true|false|unknown|entailment|contradiction|neutral|uncertain)\s*\}", text, re.IGNORECASE)
    if generic:
        label = extract_label(generic[-1])
        if label in {"true", "false", "unknown"}:
            return label
    label = extract_label(text)
    if label in {"true", "false", "unknown"}:
        return label
    final_mentions = re.findall(r"\b(?:answer|conclusion|therefore|thus|so)\b[^\n.{}]*(true|false|unknown|entailment|contradiction|neutral|uncertain)\b", text, re.IGNORECASE)
    if final_mentions:
        label = extract_label(final_mentions[-1])
        if label in {"true", "false", "unknown"}:
            return label
    trailing_mentions = re.findall(r"\b(true|false|unknown)\b", text[-800:], re.IGNORECASE)
    if trailing_mentions:
        label = extract_label(trailing_mentions[-1])
        if label in {"true", "false", "unknown"}:
            return label
    return "invalid"


def _symbcot_counterfactual_diagnostics(
    example: Example | CounterfactualExample,
    initial_label: str,
    initial_premises: tuple[str, ...],
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: SymbCoTConfig,
) -> list[dict[str, str]]:
    if not isinstance(example, Example):
        return []
    diagnostics = []
    claimed = set(initial_premises)
    cfs = generate_counterfactuals([example], seed=31, max_per_example=config.max_counterfactuals)
    for cf in cfs:
        raw = call_llm(_symbcot_counterfactual_prompt(cf, initial_label, initial_premises), min(config.max_tokens, 512), True, None)
        pred = extract_label(raw)
        failure = _counterfactual_failure(cf, initial_label, pred, claimed)
        diagnostics.append(
            {
                "probe_type": "structured_counterfactual",
                "counterfactual_id": cf.id,
                "cf_family": cf.cf_family,
                "expected_relation": cf.expected_relation,
                "changed_ids": ",".join(cf.changed_ids),
                "expected_answer": cf.label,
                "probe_answer": pred,
                "failure_type": failure,
            }
        )
    return diagnostics


def _should_repair_symbcot_cage(initial_label: str, diagnostics: list[dict[str, str]]) -> bool:
    return initial_label not in {"true", "false", "unknown"}


def _counterfactual_failure(cf, initial_label: str, pred: str, claimed: set[str]) -> str:
    if pred == "invalid":
        return "invalid_probe"
    changed = set(cf.changed_ids)
    if cf.conflict_label or cf.cf_family == "contradiction_injection":
        return "missed_contradiction" if pred != "unknown" else "ok"
    if cf.expected_relation == "changed":
        if pred == initial_label:
            return "insensitive_to_claimed_support" if changed & claimed else "missed_proof_break"
        return "ok"
    if cf.expected_relation == "preserved":
        if pred != initial_label:
            return "distractor_sensitive" if not (changed & claimed) else "unstable_claimed_support"
        return "ok"
    return "ok"


def _symbcot_counterfactual_prompt(cf: CounterfactualExample, initial_label: str, initial_premises: tuple[str, ...]) -> str:
    return f"""You are checking a SymbCoT answer with a counterfactual attribution probe.

Original SymbCoT answer: {initial_label}
Claimed support IDs: {', '.join(initial_premises) if initial_premises else 'none'}
Counterfactual changed IDs: {', '.join(cf.changed_ids)}
Expected relation to original answer: {cf.expected_relation}

Counterfactual problem:
{cf.text}

Return valid JSON only:
{{"answer": "true | false | unknown", "brief_explanation": "one short sentence"}}
"""


def _symbcot_cage_repair_prompt(
    example: Example | CounterfactualExample,
    initial_raw: str,
    initial_label: str,
    initial_premises: tuple[str, ...],
    diagnostics: list[dict[str, str]],
) -> str:
    failed = [d for d in diagnostics if d["failure_type"] != "ok"]
    diagnostic_text = "\n".join(
        f"- {d['failure_type']}: {d['cf_family']} changed {d['changed_ids']}; expected {d['expected_relation']}; probe answer {d['probe_answer']}"
        for d in failed
    )
    return f"""You are the CAGE repair layer on top of a SymbCoT logical reasoning pipeline.

Keep the SymbCoT final answer unless the counterfactual attribution diagnostics give concrete evidence that it is unsupported or unstable.

Problem:
{example.text}

SymbCoT output:
{initial_raw}

SymbCoT final answer: {initial_label}
SymbCoT claimed support IDs: {', '.join(initial_premises) if initial_premises else 'none'}

Failed diagnostics:
{diagnostic_text}

Return valid JSON only:
{{
  "answer": "true | false | unknown",
  "causal_premises": ["premise ids necessary for the repaired answer"],
  "brief_explanation": "one short sentence"
}}
"""


def _folio_transfer_diagnostics(
    example: FolioExample,
    initial_raw: str,
    initial_label: str,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: SymbCoTConfig,
) -> list[dict[str, str]]:
    probes = [
        ("nl_only", _folio_probe_prompt(example, initial_raw, initial_label, include_nl=True, include_fol=False)),
        ("fol_only", _folio_probe_prompt(example, initial_raw, initial_label, include_nl=False, include_fol=True)),
        ("combined_verifier", _folio_probe_prompt(example, initial_raw, initial_label, include_nl=True, include_fol=True)),
    ]
    diagnostics = []
    for probe_type, prompt in probes:
        raw = call_llm(prompt, min(config.max_tokens, 512), True, None)
        pred = extract_label(raw)
        diagnostics.append({"probe_type": probe_type, "probe_answer": pred, "status": _folio_probe_status(initial_label, pred), "raw": raw})
    return diagnostics


def _folio_probe_prompt(example: FolioExample, initial_raw: str, initial_label: str, include_nl: bool, include_fol: bool) -> str:
    parts = []
    if include_nl:
        parts.append("Natural-language premises:\n" + "\n".join(f"P{i + 1}: {p}" for i, p in enumerate(example.premises)))
        parts.append(f"Conclusion: {example.conclusion}")
    if include_fol:
        parts.append("Formal representation:\n" + "\n".join(f"FOL{i + 1}: {p}" for i, p in enumerate(example.premises_fol)))
        parts.append(f"Conclusion-FOL: {example.conclusion_fol}")
    return f"""Verify a SymbCoT answer for FOLIO using this transfer probe. These probes are weak consistency checks, not gold attribution labels.

Probe view:
{chr(10).join(parts)}

SymbCoT answer: {initial_label}
SymbCoT output:
{initial_raw}

Return valid JSON only:
{{"answer": "true | false | unknown", "brief_explanation": "one short sentence"}}
"""


def _folio_probe_status(initial_label: str, pred: str) -> str:
    if pred == "invalid":
        return "invalid"
    if pred == "unknown" or pred == initial_label:
        return "ok"
    return "disagree"


def _folio_symbcot_cage_repair_prompt(example: FolioExample, initial_raw: str, initial_label: str, diagnostics: list[dict[str, str]]) -> str:
    diagnostic_text = "\n".join(f"- {d['probe_type']}: answer={d['probe_answer']}, status={d['status']}" for d in diagnostics)
    return f"""You are the CAGE transfer repair layer on top of SymbCoT for FOLIO.

The probes are consistency checks only, not gold support annotations. Keep the SymbCoT answer unless the probes clearly support a different label.

Natural-language problem:
{_folio_context(example)}

SymbCoT answer: {initial_label}
SymbCoT output:
{initial_raw}

Diagnostics:
{diagnostic_text}

Return valid JSON only:
{{"answer": "true | false | unknown", "causal_premises": ["P1"], "brief_explanation": "one short sentence"}}
"""


def _folio_context(example: FolioExample) -> str:
    return "\n".join(f"P{i + 1}: {p}" for i, p in enumerate(example.premises)) + f"\nConclusion: {example.conclusion}"


def _safe_json(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _raw_row(example_id: str, method: str, raw: str) -> dict[str, str]:
    return {"example_id": example_id, "method": method, "raw_response": raw}
