from __future__ import annotations

from collections import Counter, defaultdict

from .mpss import mpss_predictions
from .prover import prove
from .schema import Example, Fact, Literal, Prediction, Rule


def symbolic_predictions(examples: list[Example], method: str = "symbolic_oracle") -> list[Prediction]:
    return [Prediction(ex.id, method, ex.label, ex.support_ids, "oracle") for ex in examples]


def train_majority_predictions(examples: list[Example], train_examples: list[Example], method: str = "train_majority") -> list[Prediction]:
    label = _majority_label(train_examples or examples)
    return [Prediction(ex.id, method, label, (), "train_majority") for ex in examples]


def depth_majority_predictions(examples: list[Example], train_examples: list[Example], method: str = "depth_majority") -> list[Prediction]:
    by_depth: dict[int, Counter[str]] = defaultdict(Counter)
    overall = Counter[str]()
    for ex in train_examples or examples:
        by_depth[ex.depth][ex.label] += 1
        overall[ex.label] += 1
    fallback = _counter_majority(overall)
    return [Prediction(ex.id, method, _counter_majority(by_depth.get(ex.depth, Counter())) or fallback, (), "depth_majority") for ex in examples]


def fact_lookup_predictions(examples: list[Example], method: str = "fact_lookup") -> list[Prediction]:
    preds: list[Prediction] = []
    for ex in examples:
        fact_by_literal = {fact.literal: fact.id for fact in ex.facts}
        if ex.query in fact_by_literal:
            preds.append(Prediction(ex.id, method, "true", (fact_by_literal[ex.query],), "fact_lookup"))
        elif ex.query.negate() in fact_by_literal:
            preds.append(Prediction(ex.id, method, "false", (fact_by_literal[ex.query.negate()],), "fact_lookup"))
        else:
            preds.append(Prediction(ex.id, method, "unknown", (), "fact_lookup"))
    return preds


def one_step_rule_predictions(examples: list[Example], method: str = "one_step_rule") -> list[Prediction]:
    preds: list[Prediction] = []
    for ex in examples:
        one_hop_rules = [rule for rule in ex.rules if len(rule.antecedents) <= 2]
        result = _prove_one_step(ex.facts, one_hop_rules, ex.query)
        preds.append(Prediction(ex.id, method, result[0], result[1], "one_step_rule"))
    return preds


def lexical_overlap_predictions(examples: list[Example], method: str = "lexical_overlap") -> list[Prediction]:
    preds: list[Prediction] = []
    for ex in examples:
        query_terms = _literal_terms(ex.query)
        scored: list[tuple[int, str, str]] = []
        for fact in ex.facts:
            scored.append((_overlap(query_terms, _literal_terms(fact.literal)), fact.id, _literal_label_for_query(fact.literal, ex.query)))
        for rule in ex.rules:
            terms = _literal_terms(rule.consequent)
            for ant in rule.antecedents:
                terms |= _literal_terms(ant)
            scored.append((_overlap(query_terms, terms), rule.id, _literal_label_for_query(rule.consequent, ex.query)))
        scored = [item for item in scored if item[0] > 0]
        if not scored:
            preds.append(Prediction(ex.id, method, "unknown", (), "lexical_overlap"))
            continue
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score = scored[0][0]
        best = [item for item in scored if item[0] == best_score]
        labels = Counter(label for _, _, label in best if label != "unknown")
        label = _counter_majority(labels) or "unknown"
        premise_ids = tuple(item[1] for item in best[:3])
        preds.append(Prediction(ex.id, method, label, premise_ids, "lexical_overlap"))
    return preds


def offline_predictions(examples: list[Example], train_examples: list[Example] | None = None) -> list[Prediction]:
    train_examples = train_examples or examples
    return (
        train_majority_predictions(examples, train_examples)
        + depth_majority_predictions(examples, train_examples)
        + fact_lookup_predictions(examples)
        + one_step_rule_predictions(examples)
        + lexical_overlap_predictions(examples)
        + mpss_predictions(examples)
        + symbolic_predictions(examples)
    )


def _majority_label(examples: list[Example]) -> str:
    counts = Counter(ex.label for ex in examples)
    return _counter_majority(counts) or "unknown"


def _counter_majority(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _prove_one_step(facts: list[Fact], rules: list[Rule], query: Literal) -> tuple[str, tuple[str, ...]]:
    fact_literals = {fact.literal: (fact.id,) for fact in facts}
    entailed = dict(fact_literals)
    for rule in rules:
        if all(ant in fact_literals for ant in rule.antecedents):
            support = tuple(sorted({rule.id, *(sid for ant in rule.antecedents for sid in fact_literals[ant])}))
            entailed.setdefault(rule.consequent, support)
    positive = entailed.get(query)
    negative = entailed.get(query.negate())
    if positive and negative:
        return "unknown", tuple(sorted(set(positive) | set(negative)))
    if positive:
        return "true", positive
    if negative:
        return "false", negative
    return "unknown", ()


def _literal_terms(literal: Literal) -> set[str]:
    return set(literal.predicate.split("__")) | set(literal.entity.split("_"))


def _overlap(left: set[str], right: set[str]) -> int:
    return len({term for term in left & right if term and term != "x"})


def _literal_label_for_query(literal: Literal, query: Literal) -> str:
    if literal == query:
        return "true"
    if literal == query.negate():
        return "false"
    if literal.predicate == query.predicate and literal.entity == query.entity:
        return "false" if literal.negated else "true"
    return "unknown"
