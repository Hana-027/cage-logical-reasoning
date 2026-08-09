from __future__ import annotations

from .schema import Fact, Label, Literal, ProofResult, Rule


def _instantiate_literal(literal: Literal, entity: str) -> Literal:
    if literal.entity == "x":
        return Literal(literal.predicate, entity, literal.negated)
    return literal


def _instantiate_rule(rule: Rule, entity: str) -> Rule:
    return Rule(
        rule.id,
        tuple(_instantiate_literal(ant, entity) for ant in rule.antecedents),
        _instantiate_literal(rule.consequent, entity),
    )


def _rule_instances(rule: Rule, entities: set[str]) -> list[Rule]:
    if rule.consequent.entity == "x" or any(ant.entity == "x" for ant in rule.antecedents):
        return [_instantiate_rule(rule, entity) for entity in sorted(entities)]
    return [rule]


def prove(facts: list[Fact], rules: list[Rule], query: Literal) -> ProofResult:
    entailed: dict[Literal, tuple[frozenset[str], int]] = {}
    entities = {fact.literal.entity for fact in facts} | {query.entity}

    for fact in facts:
        support = frozenset([fact.id])
        existing = entailed.get(fact.literal)
        if existing is None or len(support) < len(existing[0]):
            entailed[fact.literal] = (support, 0)

    changed = True
    while changed:
        changed = False
        for rule_template in rules:
            for rule in _rule_instances(rule_template, entities):
                if all(ant in entailed for ant in rule.antecedents):
                    support: frozenset[str] = frozenset([rule.id])
                    depth = 0
                    for ant in rule.antecedents:
                        ant_support, ant_depth = entailed[ant]
                        support = support | ant_support
                        depth = max(depth, ant_depth + 1)
                    existing = entailed.get(rule.consequent)
                    if existing is None or (depth, len(support), sorted(support)) < (existing[1], len(existing[0]), sorted(existing[0])):
                        entailed[rule.consequent] = (support, depth)
                        changed = True

    positive = entailed.get(query)
    negative = entailed.get(query.negate())

    if positive and negative:
        label: Label = "ambiguous"
        support_ids = tuple(sorted(positive[0] | negative[0]))
        depth = max(positive[1], negative[1])
    elif positive:
        label = "true"
        support_ids = tuple(sorted(positive[0]))
        depth = positive[1]
    elif negative:
        label = "false"
        support_ids = tuple(sorted(negative[0]))
        depth = negative[1]
    else:
        label = "unknown"
        support_ids = ()
        depth = 0

    return ProofResult(label=label, support_ids=support_ids, depth=depth, entailed=entailed)


def answer_label(facts: list[Fact], rules: list[Rule], query: Literal) -> Label:
    return prove(facts, rules, query).label
