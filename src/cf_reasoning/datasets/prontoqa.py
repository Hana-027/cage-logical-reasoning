from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..generator import render_context
from ..prover import prove
from ..schema import Example, Fact, Literal, Rule
from .proofwriter import ProofWriterFailure, ProofWriterLoadReport

_DETERMINERS = ("a ", "an ", "the ")
_RULE_SINGLE_RE = re.compile(r"^(?:Every|Each)\s+(.+?)\s+is\s+(.+?)\.?$", re.IGNORECASE)
_RULE_PLURAL_RE = re.compile(r"^(.+?)s\s+are\s+(.+?)\.?$", re.IGNORECASE)
_RULE_EVERYTHING_RE = re.compile(r"^Everything that is\s+(.+?)\s+is\s+(.+?)\.?$", re.IGNORECASE)
_FACT_RE = re.compile(r"^([A-Z][A-Za-z]*)\s+is\s+(.+?)\.?$", re.IGNORECASE)
_QUERY_RE = re.compile(r"^Prove:\s*([A-Z][A-Za-z]*)\s+is\s+(.+?)\.?$", re.IGNORECASE)
_PRONTOQA2_QUESTION_RE = re.compile(r"^Is the following statement true or false\?\s*(.+?)\.?$", re.IGNORECASE)
_STATEMENT_RE = re.compile(r"^([A-Z][A-Za-z]*)\s+is\s+(not\s+)?(.+?)\.?$", re.IGNORECASE)
_LOGIC_ATOM_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\((\$x|[A-Za-z][A-Za-z0-9_]*),\s*(True|False)\)$")
_LOGIC_RULE_RE = re.compile(r"^(.+?)\s*>>>\s*(.+?)$")


def load_prontoqa_examples(path: str | Path, limit: int | None = None, split: str = "prontoqa") -> tuple[list[Example], ProofWriterLoadReport]:
    path = Path(path)
    rows = list(_iter_pronto_rows(path))
    examples: list[Example] = []
    failures: list[ProofWriterFailure] = []
    reason_counts: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        if limit is not None and len(examples) >= limit:
            break
        parsed, failure = _parse_pronto_row(row, len(examples) + 1, row_index, split)
        if parsed is not None:
            examples.append(parsed)
            depth_counts[parsed.depth] += 1
            label_counts[parsed.label] += 1
        elif failure is not None:
            failures.append(failure)
            reason_counts[failure.reason] += 1

    loaded = len(examples) + len(failures) if limit is not None else len(rows)
    report = ProofWriterLoadReport(
        loaded=loaded,
        parsed=len(examples),
        skipped=len(failures),
        source=str(path),
        missing_fields=reason_counts["missing_fields"],
        unparsed_theory=reason_counts["unparsed_theory"],
        unparsed_query=reason_counts["unparsed_query"],
        label_mismatch=reason_counts["label_mismatch"],
        ambiguous=reason_counts["ambiguous"],
        unsupported_sentence=reason_counts["unsupported_sentence"],
        label_counts=dict(label_counts),
        depth_counts=dict(depth_counts),
        failures=tuple(failures),
    )
    return examples, report


def _iter_pronto_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        for file in sorted(path.glob("*.json")):
            yield from _iter_json_file(file)
        for file in sorted(path.glob("*.zip")):
            yield from _iter_pronto_rows(file)
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            grouped: list[list[dict[str, Any]]] = []
            for name in sorted(n for n in z.namelist() if n.endswith(".json")):
                data = json.loads(z.read(name))
                rows = []
                for key, item in _json_items(data):
                    test = item.get("test_example") if isinstance(item, dict) else None
                    if isinstance(test, dict):
                        rows.append({"source_file": name, "source_key": key, **test})
                if rows:
                    grouped.append(rows)
            max_len = max((len(rows) for rows in grouped), default=0)
            for index in range(max_len):
                for rows in grouped:
                    if index < len(rows):
                        yield rows[index]
        return
    yield from _iter_json_file(path)


def _iter_json_file(path: Path) -> Iterable[dict[str, Any]]:
    data = json.load(path.open(encoding="utf-8"))
    for key, item in _json_items(data):
        test = item.get("test_example") or item if isinstance(item, dict) else item
        if isinstance(test, dict):
            yield {"source_file": path.name, "source_key": key, **test}


def _json_items(data: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                yield str(key), value
    elif isinstance(data, list):
        for index, value in enumerate(data):
            if isinstance(value, dict):
                yield str(index), value


def _parse_pronto_row(row: dict[str, Any], idx: int, row_index: int, split: str) -> tuple[Example | None, ProofWriterFailure | None]:
    if _is_prontoqa2_row(row):
        return _parse_prontoqa2_row(row, idx, row_index, split)
    return _parse_legacy_pronto_row(row, idx, row_index, split)


def _is_prontoqa2_row(row: dict[str, Any]) -> bool:
    return "context" in row and "answer" in row and isinstance(row.get("raw_logic_programs"), list)


def _parse_legacy_pronto_row(row: dict[str, Any], idx: int, row_index: int, split: str) -> tuple[Example | None, ProofWriterFailure | None]:
    theory = str(row.get("question") or "")
    query_text = str(row.get("query") or "")
    if not theory or not query_text:
        return None, ProofWriterFailure(row_index, "missing_fields", query_text, "true")
    facts, rules, unsupported = _parse_theory(theory)
    query_info = _parse_query(query_text)
    if unsupported:
        return None, ProofWriterFailure(row_index, "unsupported_sentence", query_text, "true", unsupported[0])
    if not facts:
        return None, ProofWriterFailure(row_index, "unparsed_theory", query_text, "true")
    if query_info is None:
        return None, ProofWriterFailure(row_index, "unparsed_query", query_text, "true", query_text)
    query, query_parts = query_info
    if len(query_parts) > 1:
        rules.append(Rule(f"R{len(rules) + 1}", tuple(Literal(part, query.entity) for part in query_parts), query))
    result = prove(facts, rules, query)
    if result.label == "ambiguous":
        return None, ProofWriterFailure(row_index, "ambiguous", query_text, "true")
    if result.label != "true":
        return None, ProofWriterFailure(row_index, "label_mismatch", query_text, "true")
    row_id = f"{row.get('source_file','pronto')}_{row.get('source_key', idx)}"
    return Example(
        id=f"pronto_{_safe_id(row_id, idx)}",
        facts=facts,
        rules=rules,
        query=query,
        label="true",
        support_ids=result.support_ids,
        depth=_extract_depth(row, result.depth),
        text=render_context(facts, rules, query),
        split=split,
    ), None


def _parse_prontoqa2_row(row: dict[str, Any], idx: int, row_index: int, split: str) -> tuple[Example | None, ProofWriterFailure | None]:
    context = str(row.get("context") or "")
    question = str(row.get("question") or "")
    label = _extract_prontoqa2_label(row)
    program = _first_logic_program(row)
    if not context or not question or label is None or not program:
        return None, ProofWriterFailure(row_index, "missing_fields", question, str(row.get("answer", "")))
    parsed_logic = _parse_logic_program(program)
    if parsed_logic is None:
        return None, ProofWriterFailure(row_index, "unparsed_theory", question, label, program[:200])
    facts, rules = parsed_logic
    query = _parse_truth_question(question)
    if query is None:
        return None, ProofWriterFailure(row_index, "unparsed_query", question, label, question)
    result = prove(facts, rules, query)
    if result.label == "ambiguous":
        return None, ProofWriterFailure(row_index, "ambiguous", question, label)
    if result.label != label:
        return None, ProofWriterFailure(row_index, "label_mismatch", question, label, f"prover={result.label}")
    row_id = str(row.get("id") or f"prontoqa2_{idx}")
    return Example(
        id=f"pronto_{_safe_id(row_id, idx)}",
        facts=facts,
        rules=rules,
        query=query,
        label=label,
        support_ids=result.support_ids,
        depth=_extract_depth(row, result.depth),
        text=render_context(facts, rules, query),
        split=split,
    ), None


def _extract_prontoqa2_label(row: dict[str, Any]) -> str | None:
    answer = str(row.get("answer") or "").strip().upper()
    if answer == "A":
        return "true"
    if answer == "B":
        return "false"
    return None


def _first_logic_program(row: dict[str, Any]) -> str:
    programs = row.get("raw_logic_programs")
    if isinstance(programs, list) and programs:
        return str(programs[0])
    return ""


def _parse_logic_program(program: str) -> tuple[list[Fact], list[Rule]] | None:
    sections = _logic_sections(program)
    fact_lines = sections.get("Facts", [])
    rule_lines = sections.get("Rules", [])
    facts: list[Fact] = []
    rules: list[Rule] = []
    for line in fact_lines:
        literal = _logic_literal(line)
        if literal is None:
            return None
        facts.append(Fact(f"F{len(facts) + 1}", literal))
    for line in rule_lines:
        match = _LOGIC_RULE_RE.match(line)
        if not match:
            return None
        antecedent = _logic_literal(match.group(1))
        consequent = _logic_literal(match.group(2))
        if antecedent is None or consequent is None:
            return None
        rules.append(Rule(f"R{len(rules) + 1}", (antecedent,), consequent))
    return (facts, rules) if facts else None


def _logic_sections(program: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in program.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"Predicates:", "Facts:", "Rules:", "Query:"}:
            current = line.rstrip(":")
            sections.setdefault(current, [])
            continue
        if current in {"Facts", "Rules", "Query"}:
            sections.setdefault(current, []).append(line)
    return sections


def _logic_literal(atom: str) -> Literal | None:
    match = _LOGIC_ATOM_RE.match(atom.strip())
    if not match:
        return None
    predicate, entity, truth = match.groups()
    return Literal(
        _singularize(_normalize_predicate(predicate)),
        "x" if entity == "$x" else _normalize_entity(entity),
        negated=truth == "False",
    )


def _parse_truth_question(text: str) -> Literal | None:
    match = _PRONTOQA2_QUESTION_RE.match(text.strip())
    if not match:
        return None
    return _parse_statement_literal(match.group(1))


def _parse_statement_literal(text: str) -> Literal | None:
    match = _STATEMENT_RE.match(text.strip().rstrip("."))
    if not match:
        return None
    entity, negation, predicate = match.groups()
    preds = _parse_predicates(predicate)
    if len(preds) != 1:
        return None
    return Literal(preds[0], _normalize_entity(entity), negated=bool(negation))


def _parse_theory(text: str) -> tuple[list[Fact], list[Rule], tuple[str, ...]]:
    facts: list[Fact] = []
    rules: list[Rule] = []
    unsupported: list[str] = []
    fact_i = 1
    rule_i = 1
    for sentence in re.split(r"(?<=\.)\s+", text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        parsed_rules = _parse_rule(sentence, f"R{rule_i}")
        if parsed_rules:
            rules.extend(parsed_rules)
            rule_i += len(parsed_rules)
            continue
        parsed_facts = _parse_fact(sentence, f"F{fact_i}")
        if parsed_facts:
            facts.extend(parsed_facts)
            fact_i += len(parsed_facts)
            continue
        unsupported.append(sentence)
    return facts, rules, tuple(unsupported)


def _parse_rule(text: str, rule_id: str) -> list[Rule]:
    normalized = text.strip().rstrip(".")
    for pattern in (_RULE_EVERYTHING_RE, _RULE_SINGLE_RE, _RULE_PLURAL_RE):
        match = pattern.match(normalized)
        if not match:
            continue
        lhs, rhs = match.groups()
        antecedents = tuple(Literal(pred, "x") for pred in _parse_predicates(lhs))
        consequents = _parse_predicates(rhs)
        if not antecedents or not consequents:
            return []
        base = int(rule_id.removeprefix("R"))
        return [Rule(f"R{base + i}", antecedents, Literal(consequent, "x")) for i, consequent in enumerate(consequents)]
    return []


def _parse_fact(text: str, first_id: str) -> list[Fact]:
    match = _FACT_RE.match(text.strip().rstrip("."))
    if not match:
        return []
    entity, predicates = match.groups()
    preds = _parse_predicates(predicates)
    base = int(first_id.removeprefix("F"))
    return [Fact(f"F{base + i}", Literal(pred, _normalize_entity(entity))) for i, pred in enumerate(preds)]


def _parse_query(text: str) -> tuple[Literal, list[str]] | None:
    match = _QUERY_RE.match(text.strip().rstrip("."))
    if not match:
        return None
    entity, predicates = match.groups()
    preds = _parse_predicates(predicates)
    if not preds:
        return None
    entity_norm = _normalize_entity(entity)
    query = Literal("__and__".join(preds), entity_norm) if len(preds) > 1 else Literal(preds[0], entity_norm)
    return query, preds


def _parse_predicates(text: str) -> list[str]:
    normalized = text.strip().rstrip(".").lower()
    normalized = re.sub(r"^(?:a|an)\s+", "", normalized)
    parts = re.split(r"\s+and\s+", normalized)
    preds: list[str] = []
    for part in parts:
        part = part.strip().strip(",")
        for det in _DETERMINERS:
            if part.startswith(det):
                part = part[len(det):]
        if part:
            preds.append(_singularize(_normalize_predicate(part)))
    return preds


def _normalize_entity(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_")


def _normalize_predicate(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_")


def _singularize(text: str) -> str:
    if text.endswith("ies"):
        return text[:-3] + "y"
    if text.endswith("ses") and len(text) > 4:
        return text[:-2]
    if text.endswith("us"):
        return text
    if text.endswith("s") and len(text) > 3:
        return text[:-1]
    return text


def _extract_depth(row: dict[str, Any], fallback: int) -> int:
    match = re.search(r"(\d+)hop", str(row.get("source_file", "")))
    if match:
        return int(match.group(1))
    return fallback


def _safe_id(row_id: str, idx: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", row_id).strip("_")
    return cleaned or f"{idx:05d}"
