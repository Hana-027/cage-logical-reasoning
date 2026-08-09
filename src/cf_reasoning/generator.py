from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from .prover import prove
from .schema import Example, Fact, Literal, Rule

ENTITIES = ["alice", "bob", "charlie", "diana", "erin", "frank"]
PREDICATES = [
    "kind",
    "smart",
    "quiet",
    "brave",
    "furry",
    "blue",
    "red",
    "green",
    "young",
    "round",
    "cold",
    "happy",
]


def render_context(facts: list[Fact], rules: list[Rule], query: Literal) -> str:
    lines = [fact.to_text() for fact in facts]
    lines.extend(rule.to_text() for rule in rules)
    query_text = query.to_text().removesuffix(".")
    lines.append(f"Query: Is it true that {query_text}?")
    return "\n".join(lines)


def _make_chain(rng: random.Random, idx: int, max_depth: int) -> Example | None:
    entity = rng.choice(ENTITIES)
    depth = rng.randint(0, max_depth)
    chain_len = max(1, depth)
    preds = rng.sample(PREDICATES, chain_len + 1)
    facts: list[Fact] = [Fact("F1", Literal(preds[0], entity))]
    rules: list[Rule] = []

    for i in range(chain_len):
        rules.append(Rule(f"R{i + 1}", (Literal(preds[i], entity),), Literal(preds[i + 1], entity)))

    distractor_count = rng.randint(2, 5)
    used = {(preds[0], entity, False)}
    next_fact = 2
    for _ in range(distractor_count):
        pred = rng.choice(PREDICATES)
        ent = rng.choice(ENTITIES)
        neg = rng.random() < 0.35
        key = (pred, ent, neg)
        if key in used or Literal(pred, ent, neg) in [f.literal for f in facts]:
            continue
        facts.append(Fact(f"F{next_fact}", Literal(pred, ent, neg)))
        next_fact += 1
        used.add(key)

    label_choice = rng.random()
    if label_choice < 0.65:
        query = Literal(preds[-1], entity)
    elif label_choice < 0.82:
        query = Literal(preds[-1], entity, True)
    else:
        query = Literal(rng.choice([p for p in PREDICATES if p not in preds]), entity)

    result = prove(facts, rules, query)
    if result.label == "ambiguous":
        return None
    return Example(
        id=f"ex_{idx:05d}",
        facts=facts,
        rules=rules,
        query=query,
        label=result.label,
        support_ids=result.support_ids,
        depth=result.depth,
        text=render_context(facts, rules, query),
    )


def generate_examples(n_examples: int, seed: int = 42, max_depth: int = 3) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    attempts = 0
    while len(examples) < n_examples and attempts < n_examples * 20:
        attempts += 1
        ex = _make_chain(rng, len(examples) + 1, max_depth)
        if ex is not None:
            examples.append(ex)
    if len(examples) < n_examples:
        raise RuntimeError(f"Only generated {len(examples)} valid examples after {attempts} attempts")
    return examples


def _encode(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(type(obj).__name__)


def write_jsonl(path: str | Path, rows: Iterable[object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_encode(row), ensure_ascii=False) + "\n")
