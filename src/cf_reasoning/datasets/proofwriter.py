from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..generator import render_context
from ..prover import prove
from ..schema import Example, Fact, Literal, Rule


@dataclass(frozen=True)
class ProofWriterFailure:
    row_index: int
    reason: str
    question: str = ""
    answer: str = ""
    failing_sentence: str = ""


@dataclass(frozen=True)
class ProofWriterLoadReport:
    loaded: int
    parsed: int
    skipped: int
    source: str
    missing_fields: int = 0
    unparsed_theory: int = 0
    unparsed_query: int = 0
    label_mismatch: int = 0
    ambiguous: int = 0
    unsupported_sentence: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    depth_counts: dict[int, int] = field(default_factory=dict)
    failures: tuple[ProofWriterFailure, ...] = ()

    def to_dict(self, split: str = "") -> dict[str, Any]:
        return {
            "split": split,
            "source": self.source,
            "loaded": self.loaded,
            "parsed": self.parsed,
            "skipped": self.skipped,
            "coverage": self.parsed / self.loaded if self.loaded else 0.0,
            "missing_fields": self.missing_fields,
            "unparsed_theory": self.unparsed_theory,
            "unparsed_query": self.unparsed_query,
            "label_mismatch": self.label_mismatch,
            "ambiguous": self.ambiguous,
            "unsupported_sentence": self.unsupported_sentence,
            "label_counts": json.dumps(self.label_counts, ensure_ascii=False, sort_keys=True),
            "depth_counts": json.dumps(self.depth_counts, ensure_ascii=False, sort_keys=True),
        }


@dataclass(frozen=True)
class ParsedTheory:
    facts: list[Fact]
    rules: list[Rule]
    unsupported: tuple[str, ...]


_FACT_RE = re.compile(r"^(.+?)\s+(?:is|are)\s+(not\s+)?([a-z][A-Za-z0-9_-]*)\.?$", re.IGNORECASE)
_RELATION_RE = re.compile(r"^(.+?)\s+([a-z][A-Za-z0-9_-]*)\s+(.+?)\.?$", re.IGNORECASE)
_RULE_RE = re.compile(r"^if\s+(.+?)\s+then\s+(.+?)\.?$", re.IGNORECASE)
_GENERIC_RULE_RE = re.compile(r"^(?:all\s+)?(.+?)\s+(?:things|people)\s+are\s+(not\s+)?([a-z][A-Za-z0-9_-]*)\.?$", re.IGNORECASE)
_ID_LINE_RE = re.compile(r"^\s*([A-Za-z]+\d+)\s*[:\.]\s*(.+?)\s*$")
_TRANSLATION_PART_RE = re.compile(r"\$(answer|proof|question|context)\$\s*=\s*(.*?)(?=\s*;\s*\$(?:answer|proof|question|context)\$\s*=|$)", re.IGNORECASE)
_LABEL_MAP = {
    "true": "true",
    "false": "false",
    "unknown": "unknown",
    "unk": "unknown",
}
_PRONOUNS = {"someone", "something", "thing", "things", "it", "they"}
_DETERMINERS = ("the ", "a ", "an ")
_RELATION_VERBS = {"chase", "chases", "eat", "eats", "like", "likes", "need", "needs", "see", "sees", "visit", "visits"}


def load_proofwriter_examples(path: str | Path, limit: int | None = None, split: str = "") -> tuple[list[Example], ProofWriterLoadReport]:
    rows = list(_iter_rows(Path(path)))
    examples: list[Example] = []
    failures: list[ProofWriterFailure] = []
    reason_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()

    for row_index, row in enumerate(rows):
        if limit is not None and len(examples) >= limit:
            break
        parsed, failure = _parse_row(row, len(examples) + 1, row_index, split=split)
        if parsed is not None:
            examples.append(parsed)
            label_counts[parsed.label] += 1
            depth_counts[parsed.depth] += 1
        elif failure is not None:
            failures.append(failure)
            reason_counts[failure.reason] += 1

    loaded = len(rows) if limit is None else min(len(rows), rows.index(rows[-1]) + 1 if rows else 0)
    if limit is not None:
        loaded = min(len(rows), max(limit, len(examples) + len(failures)))
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


def write_parse_failures(path: str | Path, report: ProofWriterLoadReport, split: str = "") -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "row_index", "reason", "question", "answer", "failing_sentence"])
        writer.writeheader()
        for failure in report.failures:
            writer.writerow({"split": split, **failure.__dict__})


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"ProofWriter data path does not exist: {path}")
    if path.is_dir():
        files = sorted([*path.glob("*.jsonl"), *path.glob("*.json")])
        for file in files:
            yield from _iter_rows(file)
        return
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield from _expand_official_row(row)
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield from _expand_official_row(row)
    elif isinstance(data, dict):
        for key in ("data", "examples", "instances", "train", "validation", "test"):
            value = data.get(key)
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        yield from _expand_official_row(row)


def _expand_official_row(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    questions = row.get("questions")
    if not isinstance(questions, dict):
        yield row
        return
    theory = _official_theory_text(row)
    base_id = str(row.get("id") or "proofwriter")
    for question_id, question in sorted(questions.items(), key=lambda item: _natural_key(item[0])):
        if not isinstance(question, dict):
            continue
        expanded = dict(row)
        expanded.pop("questions", None)
        expanded["id"] = f"{base_id}_{question_id}"
        expanded["theory"] = theory
        expanded["question"] = str(question.get("question") or "")
        expanded["answer"] = question.get("answer")
        expanded["QDep"] = question.get("QDep", row.get("maxD"))
        expanded["QLen"] = question.get("QLen", "")
        expanded["strategy"] = question.get("strategy", "")
        expanded["proofs"] = question.get("proofs", "")
        expanded["representation"] = question.get("representation", "")
        yield expanded


def _official_theory_text(row: dict[str, Any]) -> str:
    triples = row.get("triples")
    rules = row.get("rules")
    if isinstance(triples, dict) or isinstance(rules, dict):
        lines: list[str] = []
        if isinstance(triples, dict):
            for key in sorted(triples, key=_natural_key):
                value = triples[key]
                if isinstance(value, dict) and isinstance(value.get("text"), str):
                    lines.append(f"{key}: {value['text']}")
        if isinstance(rules, dict):
            for key in sorted(rules, key=_natural_key):
                value = rules[key]
                if isinstance(value, dict) and isinstance(value.get("text"), str):
                    lines.append(f"{key}: {value['text']}")
        if lines:
            return "\n".join(lines)
    return _first_text(row, "theory", "context", "story", "facts", "rules", "text", "input")


def _natural_key(text: str) -> tuple[str, int]:
    match = re.match(r"^([A-Za-z_]+)(\d+)$", str(text))
    if not match:
        return str(text), 0
    return match.group(1), int(match.group(2))


def _parse_row(row: dict[str, Any], idx: int, row_index: int, split: str = "") -> tuple[Example | None, ProofWriterFailure | None]:
    row = _normalize_translation_row(row)
    text = _first_text(row, "theory", "context", "story", "facts", "rules", "text", "input")
    query_text = _first_text(row, "question", "query", "hypothesis", "statement")
    label = _extract_label(row)
    if not text or not query_text or label is None:
        return None, ProofWriterFailure(row_index, "missing_fields", query_text, str(label or ""))

    parsed_theory = _parse_theory(text)
    query = _parse_literal(query_text)
    if parsed_theory.unsupported:
        return None, ProofWriterFailure(row_index, "unsupported_sentence", query_text, label, parsed_theory.unsupported[0])
    if not parsed_theory.facts:
        return None, ProofWriterFailure(row_index, "unparsed_theory", query_text, label)
    if query is None:
        return None, ProofWriterFailure(row_index, "unparsed_query", query_text, label, query_text)

    result = prove(parsed_theory.facts, parsed_theory.rules, query)
    if result.label == "ambiguous":
        return None, ProofWriterFailure(row_index, "ambiguous", query_text, label)
    if label != result.label:
        return None, ProofWriterFailure(row_index, "label_mismatch", query_text, label)

    row_id = str(row.get("id") or row.get("example_id") or f"pw_{idx:05d}")
    return Example(
        id=f"pw_{_safe_id(row_id, idx)}",
        facts=parsed_theory.facts,
        rules=parsed_theory.rules,
        query=query,
        label=result.label,
        support_ids=result.support_ids,
        depth=_extract_depth(row, result.depth),
        text=render_context(parsed_theory.facts, parsed_theory.rules, query),
        split=split,
    ), None


def _normalize_translation_row(row: dict[str, Any]) -> dict[str, Any]:
    translation = row.get("translation")
    if not isinstance(translation, dict):
        return row
    normalized = dict(row)
    en = translation.get("en")
    ro = translation.get("ro")
    if isinstance(en, str):
        normalized.update(_parse_translation_parts(en))
    if isinstance(ro, str):
        normalized.update(_parse_translation_parts(ro))
    return normalized


def _parse_translation_parts(text: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for key, value in _TRANSLATION_PART_RE.findall(text):
        parts[key.lower()] = value.strip().rstrip(";").strip()
    return parts


def _first_text(row: dict[str, Any], *keys: str) -> str:
    parts: list[str] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            parts.extend(str(v).strip() for v in value if str(v).strip())
        elif isinstance(value, dict):
            parts.extend(str(v).strip() for v in value.values() if str(v).strip())
    return "\n".join(parts)


def _extract_label(row: dict[str, Any]) -> str | None:
    for key in ("answer", "label", "target", "gold", "gold_label"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip().lower()
        text = text.replace("entailment", "true").replace("contradiction", "false").replace("neutral", "unknown")
        if text in _LABEL_MAP:
            return _LABEL_MAP[text]
    return None


def _extract_depth(row: dict[str, Any], fallback: int) -> int:
    for key in ("depth", "QDep", "qdep", "proof_depth"):
        value = row.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return fallback


def _parse_theory(text: str) -> ParsedTheory:
    facts: list[Fact] = []
    rules: list[Rule] = []
    unsupported: list[str] = []
    fact_i = 1
    rule_i = 1
    for raw in _split_sentences(text):
        line_id = None
        line = raw.strip()
        match = _ID_LINE_RE.match(line)
        if match:
            line_id, line = match.group(1).upper(), match.group(2)
        rule = _parse_rule(line, line_id or f"R{rule_i}")
        if rule is not None:
            rules.append(rule)
            rule_i += 1
            continue
        literal = _parse_literal(line)
        if literal is not None:
            facts.append(Fact(line_id or f"F{fact_i}", literal))
            fact_i += 1
            continue
        if line:
            unsupported.append(raw.strip())
    return ParsedTheory(facts, rules, tuple(unsupported))


def _split_sentences(text: str) -> list[str]:
    chunks: list[str] = []
    text = re.sub(r"\s+(sent\d+\s*:)", r"\n\1", text)
    for line in text.replace(";", ".").splitlines():
        pieces = re.split(r"(?<=\.)\s+", line.strip())
        chunks.extend(piece.strip() for piece in pieces if piece.strip())
    return chunks


def _parse_rule(text: str, rule_id: str) -> Rule | None:
    normalized = text.strip().rstrip(".")
    match = _RULE_RE.match(normalized)
    if match:
        lhs, rhs = match.group(1), match.group(2)
        consequent = _parse_literal(rhs)
        if consequent is None:
            return None
        entity = consequent.entity
        antecedents = tuple(_parse_clause(part, entity) for part in re.split(r"\s+and\s+", lhs, flags=re.IGNORECASE))
        antecedents = tuple(lit for lit in antecedents if lit is not None)
        if not antecedents:
            return None
        return Rule(rule_id, antecedents, consequent)

    match = _GENERIC_RULE_RE.match(normalized)
    if not match:
        return None
    lhs, neg, predicate = match.groups()
    antecedents = tuple(_parse_predicate_phrase(part, "x") for part in re.split(r"\s*,\s*|\s+and\s+", lhs, flags=re.IGNORECASE))
    antecedents = tuple(lit for lit in antecedents if lit is not None)
    if not antecedents:
        return None
    return Rule(rule_id, antecedents, Literal(_normalize_predicate(predicate), "x", bool(neg)))


def _parse_clause(text: str, fallback_entity: str) -> Literal | None:
    return _parse_literal(text)


def _parse_predicate_phrase(text: str, entity: str) -> Literal | None:
    normalized = text.strip().lower()
    normalized = re.sub(r"^(all|if|something|someone|thing|things)\s+", "", normalized)
    normalized = normalized.strip()
    if not normalized:
        return None
    if normalized.startswith("not "):
        return Literal(_normalize_predicate(normalized.removeprefix("not ")), entity, True)
    return Literal(_normalize_predicate(normalized), entity)


def _parse_literal(text: str) -> Literal | None:
    normalized = text.strip().rstrip(".").replace("Is it true that ", "").replace("is it true that ", "")
    normalized = normalized.replace("?", "")
    match = _FACT_RE.match(normalized)
    if match:
        entity, neg, predicate = match.groups()
        return Literal(_normalize_predicate(predicate), _normalize_entity(entity), bool(neg))
    match = _RELATION_RE.match(normalized)
    if not match:
        return None
    relation_parts = _parse_relation_parts(normalized)
    if relation_parts is None:
        return None
    entity, relation, obj, negated = relation_parts
    entity_norm = _normalize_entity(entity)
    if entity_norm == "x" and _normalize_entity(obj) == "x":
        return None
    return Literal(f"{_normalize_predicate(relation)}__{_normalize_entity(obj)}", entity_norm, negated)


def _parse_relation_parts(text: str) -> tuple[str, str, str, bool] | None:
    tokens = text.strip().split()
    if len(tokens) < 3:
        return None
    lowered = [token.lower() for token in tokens]
    for index, token in enumerate(lowered):
        if token in _RELATION_VERBS and index > 0 and index < len(tokens) - 1:
            if index >= 2 and lowered[index - 1] == "not" and lowered[index - 2] in {"do", "does", "did"}:
                return " ".join(tokens[: index - 2]), token, " ".join(tokens[index + 1 :]), True
            if index >= 1 and lowered[index - 1] in {"do", "does", "did"}:
                return " ".join(tokens[: index - 1]), token, " ".join(tokens[index + 1 :]), False
            return " ".join(tokens[:index]), token, " ".join(tokens[index + 1 :]), False
    entity_width = 2 if tokens[0].lower() in {"the", "a", "an"} and len(tokens) >= 4 else 1
    entity = " ".join(tokens[:entity_width])
    rest = tokens[entity_width:]
    if len(rest) >= 4 and rest[0].lower() in {"do", "does", "did"} and rest[1].lower() == "not":
        return entity, rest[2], " ".join(rest[3:]), True
    if len(rest) >= 3 and rest[0].lower() in {"do", "does", "did"}:
        return entity, rest[1], " ".join(rest[2:]), False
    if len(rest) < 2:
        return None
    return entity, rest[0], " ".join(rest[1:]), False


def _normalize_entity(text: str) -> str:
    normalized = text.strip().lower().replace("-", "_")
    for determiner in _DETERMINERS:
        if normalized.startswith(determiner):
            normalized = normalized.removeprefix(determiner)
            break
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_")
    if normalized in _PRONOUNS:
        return "x"
    return normalized


def _normalize_predicate(text: str) -> str:
    normalized = text.strip().lower().replace("-", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_")
    irregular = {"chases": "chase", "sees": "see", "does": "do", "goes": "go"}
    if normalized in irregular:
        return irregular[normalized]
    if normalized.endswith("ies"):
        return normalized[:-3] + "y"
    if normalized.endswith("ses") and len(normalized) > 4:
        return normalized[:-1]
    if normalized.endswith("es") and len(normalized) > 4 and normalized[-3] in {"s", "x", "z"}:
        return normalized[:-2]
    if normalized.endswith("s") and len(normalized) > 3:
        return normalized[:-1]
    return normalized


def _safe_id(row_id: str, idx: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", row_id).strip("_")
    return cleaned or f"{idx:05d}"
