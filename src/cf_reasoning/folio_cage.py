from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable

from .datasets import FolioExample
from .schema import Prediction

CallLLM = Callable[[str, int, bool, dict | None], str]
ExtractLabel = Callable[[str], str]

FOLIO_TRANSFER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["true", "false", "unknown"]},
        "causal_premises": {"type": "array", "items": {"type": "string"}},
        "brief_explanation": {"type": "string"},
    },
    "required": ["answer", "causal_premises", "brief_explanation"],
    "additionalProperties": False,
}
FOLIO_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["true", "false", "unknown"]},
        "agreement": {"type": "string", "enum": ["agree", "disagree", "uncertain"]},
        "brief_explanation": {"type": "string"},
    },
    "required": ["answer", "agreement", "brief_explanation"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class FolioCAGEConfig:
    max_tokens: int = 512
    n_candidates: int = 3


def folio_cpa_predict(
    example: FolioExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: FolioCAGEConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or FolioCAGEConfig()
    raw = call_llm(_folio_cpa_prompt(example), config.max_tokens, True, FOLIO_TRANSFER_SCHEMA)
    return Prediction(example.id, "llm_cpa", extract_label(raw), _extract_premise_ids(raw), raw), [
        _raw_row(example.id, "llm_cpa", raw)
    ]


def folio_cage_predict(
    example: FolioExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: FolioCAGEConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or FolioCAGEConfig()
    raw_rows: list[dict[str, str]] = []
    draft_raw = call_llm(_folio_cpa_prompt(example), config.max_tokens, True, FOLIO_TRANSFER_SCHEMA)
    draft_label = extract_label(draft_raw)
    raw_rows.append(_raw_row(example.id, "llm_cage_initial", draft_raw))
    diagnostics = _folio_probe_diagnostics(example, draft_raw, draft_label, call_llm, extract_label, config)
    repair_raw = call_llm(_folio_repair_prompt(example, draft_raw, draft_label, diagnostics), config.max_tokens, True, FOLIO_TRANSFER_SCHEMA)
    repaired_label = extract_label(repair_raw)
    if repaired_label == "invalid":
        repaired_label = draft_label
    raw_rows.append(_raw_row(example.id, "llm_cage", repair_raw))
    trace = _trace(draft_raw, draft_label, diagnostics, repair_raw, repaired_label)
    return Prediction(example.id, "llm_cage", repaired_label, _extract_premise_ids(repair_raw), trace), raw_rows


def folio_cage_select_predict(
    example: FolioExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: FolioCAGEConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or FolioCAGEConfig()
    raw_rows: list[dict[str, str]] = []
    candidate_raw = call_llm(_folio_candidate_prompt(example, config.n_candidates), config.max_tokens, True, None)
    raw_rows.append(_raw_row(example.id, "llm_cage_select_candidates", candidate_raw))
    candidates = _parse_candidates(candidate_raw, extract_label)
    if not candidates:
        candidates = [(extract_label(candidate_raw), candidate_raw)]
    scored = []
    for label, raw in candidates[: config.n_candidates]:
        diagnostics = _folio_probe_diagnostics(example, raw, label, call_llm, extract_label, config)
        score = _diagnostic_score(label, diagnostics)
        scored.append((score, label, raw, diagnostics))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_label, best_raw, best_diagnostics = scored[0]
    repair_raw = call_llm(_folio_select_repair_prompt(example, best_label, best_raw, best_score, best_diagnostics), config.max_tokens, True, FOLIO_TRANSFER_SCHEMA)
    final_label = extract_label(repair_raw)
    if final_label == "invalid":
        final_label = best_label
    raw_rows.append(_raw_row(example.id, "llm_cage_select", repair_raw))
    trace = _trace(best_raw, best_label, best_diagnostics, repair_raw, final_label, selected_score=best_score)
    return Prediction(example.id, "llm_cage_select", final_label, _extract_premise_ids(repair_raw), trace), raw_rows


def _folio_probe_diagnostics(
    example: FolioExample,
    draft_raw: str,
    draft_label: str,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: FolioCAGEConfig,
) -> list[dict[str, str]]:
    probes = [
        ("nl_only", _nl_only_prompt(example)),
        ("fol_only", _fol_only_prompt(example)),
        ("negated_conclusion", _negated_conclusion_prompt(example)),
        ("verifier", _verifier_prompt(example, draft_raw, draft_label)),
    ]
    diagnostics = []
    for probe_type, prompt in probes:
        raw = call_llm(prompt, min(config.max_tokens, 512), True, FOLIO_VERIFY_SCHEMA)
        pred = extract_label(raw)
        diagnostics.append(
            {
                "probe_type": probe_type,
                "draft_answer": draft_label,
                "probe_answer": pred,
                "status": _probe_status(probe_type, draft_label, pred),
                "raw": raw,
            }
        )
    return diagnostics


def _probe_status(probe_type: str, draft_label: str, pred: str) -> str:
    if pred == "invalid":
        return "invalid"
    if probe_type == "negated_conclusion":
        if draft_label == "true" and pred == "true":
            return "conflict"
        if draft_label == "false" and pred == "false":
            return "conflict"
        return "ok"
    return "ok" if pred == draft_label or pred == "unknown" else "disagree"


def _diagnostic_score(label: str, diagnostics: list[dict[str, str]]) -> float:
    if label not in {"true", "false", "unknown"}:
        return -1.0
    if not diagnostics:
        return 0.0
    ok = sum(d["status"] == "ok" for d in diagnostics)
    invalid = sum(d["status"] == "invalid" for d in diagnostics)
    conflict = sum(d["status"] == "conflict" for d in diagnostics)
    return ok / len(diagnostics) - 0.25 * invalid - 0.50 * conflict


def _folio_cpa_prompt(example: FolioExample) -> str:
    return f"""You are solving a FOLIO logical reasoning problem.

Use the natural-language premises and formal representation. Return the answer and the premise IDs that are causally useful. Premise IDs are P1, P2, etc.; these are claimed supports, not gold annotations.

{_folio_problem(example, include_nl=True, include_fol=True)}

Return valid JSON only:
{{
  "answer": "true | false | unknown",
  "causal_premises": ["P1", "P2"],
  "brief_explanation": "one short sentence"
}}
"""


def _folio_candidate_prompt(example: FolioExample, n_candidates: int) -> str:
    return f"""Generate {n_candidates} diverse candidate answers for this FOLIO problem.

Use both natural-language and FOL representations. Each candidate must include an answer, claimed premise IDs, and a concise explanation.

{_folio_problem(example, include_nl=True, include_fol=True)}

Return valid JSON only:
{{
  "candidates": [
    {{"answer": "true | false | unknown", "causal_premises": ["P1"], "brief_explanation": "one short sentence"}}
  ]
}}
"""


def _nl_only_prompt(example: FolioExample) -> str:
    return f"""Answer this FOLIO problem using only the natural-language premises.

{_folio_problem(example, include_nl=True, include_fol=False)}

Return valid JSON only:
{{"answer": "true | false | unknown", "agreement": "agree | disagree | uncertain", "brief_explanation": "one short sentence"}}
"""


def _fol_only_prompt(example: FolioExample) -> str:
    return f"""Answer this FOLIO problem using only the formal representation.

{_folio_problem(example, include_nl=False, include_fol=True)}

Return valid JSON only:
{{"answer": "true | false | unknown", "agreement": "agree | disagree | uncertain", "brief_explanation": "one short sentence"}}
"""


def _negated_conclusion_prompt(example: FolioExample) -> str:
    return f"""Check the negated-conclusion probe for this FOLIO problem.

Original conclusion:
{example.conclusion}

Formal conclusion:
{example.conclusion_fol}

Question: Does the negation of the conclusion follow from the premises? If the original conclusion is contradicted, answer true. If the negation is contradicted, answer false. If neither side follows, answer unknown.

{_folio_problem(example, include_nl=True, include_fol=True)}

Return valid JSON only:
{{"answer": "true | false | unknown", "agreement": "agree | disagree | uncertain", "brief_explanation": "one short sentence"}}
"""


def _verifier_prompt(example: FolioExample, draft_raw: str, draft_label: str) -> str:
    return f"""Verify this candidate answer for a FOLIO logical reasoning problem.

Candidate label: {draft_label}
Candidate response:
{draft_raw}

Check whether the candidate label is supported, contradicted, or unknown under the premises. Return the corrected label if needed.

{_folio_problem(example, include_nl=True, include_fol=True)}

Return valid JSON only:
{{"answer": "true | false | unknown", "agreement": "agree | disagree | uncertain", "brief_explanation": "one short sentence"}}
"""


def _folio_repair_prompt(example: FolioExample, draft_raw: str, draft_label: str, diagnostics: list[dict[str, str]]) -> str:
    diagnostic_text = _diagnostic_text(diagnostics)
    return f"""Repair a FOLIO answer using transfer consistency diagnostics.

These probes are not gold support annotations; use them conservatively. Keep the original answer unless the probes reveal a clear contradiction or a better supported label.

Original problem:
{_folio_problem(example, include_nl=True, include_fol=True)}

Initial label: {draft_label}
Initial response:
{draft_raw}

Diagnostics:
{diagnostic_text}

Return valid JSON only:
{{
  "answer": "true | false | unknown",
  "causal_premises": ["P1", "P2"],
  "brief_explanation": "one short sentence"
}}
"""


def _folio_select_repair_prompt(example: FolioExample, label: str, raw: str, score: float, diagnostics: list[dict[str, str]]) -> str:
    return f"""Repair the selected FOLIO candidate answer using consistency diagnostics.

Selected candidate score: {score:.2f}
Selected label: {label}
Selected response:
{raw}

Problem:
{_folio_problem(example, include_nl=True, include_fol=True)}

Diagnostics:
{_diagnostic_text(diagnostics)}

Return valid JSON only:
{{
  "answer": "true | false | unknown",
  "causal_premises": ["P1", "P2"],
  "brief_explanation": "one short sentence"
}}
"""


def _folio_problem(example: FolioExample, include_nl: bool, include_fol: bool) -> str:
    parts = []
    if include_nl:
        premises = "\n".join(f"P{i + 1}: {premise}" for i, premise in enumerate(example.premises))
        parts.append(f"Natural-language premises:\n{premises}\nConclusion: {example.conclusion}")
    if include_fol:
        fol_lines = "\n".join(f"FOL{i + 1}: {premise}" for i, premise in enumerate(example.premises_fol))
        parts.append(f"Formal representation:\n{fol_lines}\nConclusion-FOL: {example.conclusion_fol}")
    return "\n\n".join(parts)


def _parse_candidates(text: str, extract_label: ExtractLabel) -> list[tuple[str, str]]:
    data = _json_from_text(text)
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        return []
    candidates = []
    for item in data["candidates"]:
        if isinstance(item, dict):
            raw = json.dumps(item, sort_keys=True)
            candidates.append((extract_label(raw), raw))
    return candidates


def _extract_premise_ids(text: str) -> tuple[str, ...]:
    data = _json_from_text(text)
    if not isinstance(data, dict):
        return ()
    values = data.get("causal_premises", [])
    if isinstance(values, list):
        return tuple(str(v) for v in values)
    return ()


def _diagnostic_text(diagnostics: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- {d['probe_type']}: answer={d['probe_answer']}, status={d['status']}"
        for d in diagnostics
    ) or "- no diagnostics"


def _trace(draft_raw: str, draft_label: str, diagnostics: list[dict[str, str]], repair_raw: str, final_label: str, selected_score: float | None = None) -> str:
    trace = {
        "draft_raw": draft_raw,
        "draft_answer": draft_label,
        "diagnostics": diagnostics,
        "repair_raw": _safe_json(repair_raw),
        "final_answer": final_label,
    }
    if selected_score is not None:
        trace["selected_score"] = selected_score
    return json.dumps(trace, sort_keys=True)


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


def _safe_json(text: str):
    data = _json_from_text(text)
    return data if data is not None else text
