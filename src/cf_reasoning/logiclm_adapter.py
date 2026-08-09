from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from .datasets import FolioExample
from .datasets import prontoqa as prontoqa_loader
from .prover import prove
from .schema import CounterfactualExample, Example, Fact, Literal, Prediction, Rule

CallLLM = Callable[[str, int, bool, dict | None], str]
ExtractLabel = Callable[[str], str]
ExtractPremises = Callable[[str], tuple[str, ...]]

ROOT = Path(__file__).resolve().parents[2]
LOGIC_LM_ROOT = ROOT / "Logic-LLM"
DATASET_ALIASES = {
    "proofwriter": "ProofWriter",
    "prontoqa": "ProntoQA",
    "folio": "FOLIO",
    "synthetic": "ProofWriter",
    "": "ProofWriter",
}

LOGICLM_FALLBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["true", "false", "unknown"]},
        "causal_premises": {"type": "array", "items": {"type": "string"}},
        "brief_explanation": {"type": "string"},
    },
    "required": ["answer", "causal_premises", "brief_explanation"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LogicLMConfig:
    max_tokens: int = 512
    prompt_root: Path = LOGIC_LM_ROOT / "models" / "prompts"


def logiclm_predict(
    example: Example | CounterfactualExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    extract_premises: ExtractPremises,
    config: LogicLMConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or LogicLMConfig()
    dataset = _dataset_name(example.split)
    context, question = _example_problem_parts(example)
    program_prompt = _program_prompt(dataset, context, question, config)
    program_raw = call_llm(program_prompt, config.max_tokens, False, None)
    raw_rows = [_raw_row(example.id, "llm_logiclm_program", program_raw)]
    solver = _execute_structured_program(program_raw)
    if solver is not None:
        label, support_ids = solver
        trace = {"dataset": dataset, "program_raw": program_raw, "solver_status": "success", "final_answer": label}
        return Prediction(example.id, "llm_logiclm", label, support_ids, json.dumps(trace, sort_keys=True)), raw_rows
    fallback_raw = call_llm(_logiclm_fallback_prompt(example, program_raw), config.max_tokens, True, LOGICLM_FALLBACK_SCHEMA)
    raw_rows.append(_raw_row(example.id, "llm_logiclm", fallback_raw))
    label = extract_label(fallback_raw)
    premises = extract_premises(fallback_raw)
    trace = {"dataset": dataset, "program_raw": program_raw, "solver_status": "fallback", "fallback_raw": _safe_json(fallback_raw), "final_answer": label}
    return Prediction(example.id, "llm_logiclm", label, premises, json.dumps(trace, sort_keys=True)), raw_rows


def folio_logiclm_predict(
    example: FolioExample,
    call_llm: CallLLM,
    extract_label: ExtractLabel,
    config: LogicLMConfig | None = None,
) -> tuple[Prediction, list[dict[str, str]]]:
    config = config or LogicLMConfig()
    context = " ".join(example.premises)
    question = f"Based on the above information, is the following statement true, false, or uncertain? {example.conclusion}"
    program_prompt = _program_prompt("FOLIO", context, question, config)
    program_raw = call_llm(program_prompt, config.max_tokens, False, None)
    raw_rows = [_raw_row(example.id, "llm_logiclm_program", program_raw)]
    fallback_raw = call_llm(_folio_logiclm_fallback_prompt(example, program_raw), config.max_tokens, True, None)
    raw_rows.append(_raw_row(example.id, "llm_logiclm", fallback_raw))
    label = extract_label(fallback_raw)
    trace = {"dataset": "FOLIO", "program_raw": program_raw, "solver_status": "fallback", "fallback_raw": _safe_json(fallback_raw), "final_answer": label}
    return Prediction(example.id, "llm_logiclm", label, (), json.dumps(trace, sort_keys=True)), raw_rows


def _program_prompt(dataset: str, context: str, question: str, config: LogicLMConfig) -> str:
    path = config.prompt_root / f"{dataset}.txt"
    if not path.exists():
        path = LOGIC_LM_ROOT / "models" / "prompts" / f"{dataset}.txt"
    template = path.read_text(encoding="utf-8")
    return template.replace("[[PROBLEM]]", context).replace("[[QUESTION]]", question)


def _execute_structured_program(program_raw: str) -> tuple[str, tuple[str, ...]] | None:
    parsed = _parse_logiclm_program(_strip_logic_comments(program_raw))
    if parsed is None:
        return None
    query = _query_from_program(program_raw)
    if query is None:
        return None
    query = _normalize_literal(query)
    facts, rules = _normalize_program_symbols(*parsed)
    result = prove(facts, rules, query)
    if result.label == "ambiguous":
        return "unknown", result.support_ids
    return result.label, result.support_ids


def _parse_logiclm_program(program: str) -> tuple[list[Fact], list[Rule]] | None:
    sections = prontoqa_loader._logic_sections(program)
    facts = []
    rules = []
    for line in sections.get("Facts", []):
        literal = prontoqa_loader._logic_literal(line.strip())
        if literal is None:
            return None
        facts.append(Fact(f"F{len(facts) + 1}", literal))
    for line in sections.get("Rules", []):
        if ">>>" not in line:
            return None
        lhs, rhs = line.split(">>>", 1)
        antecedents = [prontoqa_loader._logic_literal(part.strip()) for part in lhs.split("&&")]
        consequents = [prontoqa_loader._logic_literal(part.strip()) for part in rhs.split("&&")]
        if not antecedents or not consequents or any(item is None for item in antecedents + consequents):
            return None
        for consequent in consequents:
            rules.append(Rule(f"R{len(rules) + 1}", tuple(antecedents), consequent))
    return (facts, rules) if facts else None


def _strip_logic_comments(program_raw: str) -> str:
    lines = []
    for line in program_raw.splitlines():
        if ":::" in line:
            line = line.split(":::", 1)[0].rstrip()
        lines.append(line)
    return "\n".join(lines)


def _normalize_program_symbols(facts: list[Fact], rules: list[Rule]) -> tuple[list[Fact], list[Rule]]:
    return [Fact(fact.id, _normalize_literal(fact.literal)) for fact in facts], [
        Rule(rule.id, tuple(_normalize_literal(ant) for ant in rule.antecedents), _normalize_literal(rule.consequent))
        for rule in rules
    ]


def _normalize_literal(literal: Literal) -> Literal:
    return Literal(literal.predicate, literal.entity.lower(), literal.negated)


def _query_from_program(program_raw: str):
    sections = prontoqa_loader._logic_sections(program_raw)
    query_lines = sections.get("Query", [])
    if not query_lines:
        return None
    return prontoqa_loader._logic_literal(query_lines[0].split(":::", 1)[0].strip())


def _logiclm_fallback_prompt(example: Example | CounterfactualExample, program_raw: str) -> str:
    return f"""You are implementing the fallback path of a Logic-LM-style baseline.

The symbolic program below could not be executed by the local solver. Use the original problem and the generated symbolic program to return the final answer.

Problem:
{example.text}

Generated symbolic program:
{program_raw}

Return valid JSON only:
{{"answer": "true | false | unknown", "causal_premises": ["support IDs if available"], "brief_explanation": "one short sentence"}}
"""


def _folio_logiclm_fallback_prompt(example: FolioExample, program_raw: str) -> str:
    return f"""You are implementing the fallback path of a Logic-LM-style FOLIO baseline.

Use the generated symbolic formulation as the main representation, but check it against the natural-language and FOL premises before returning the final label.

Natural-language premises:
{example.text}

FOL premises:
{chr(10).join(example.premises_fol)}
Conclusion-FOL: {example.conclusion_fol}

Generated symbolic program:
{program_raw}

Return valid JSON only:
{{"answer": "true | false | unknown", "brief_explanation": "one short sentence"}}
"""


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


def _safe_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _raw_row(example_id: str, method: str, raw: str) -> dict[str, str]:
    return {"example_id": example_id, "method": method, "raw_response": raw}
