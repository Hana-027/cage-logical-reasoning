from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .prover import prove
from .schema import Example, Fact, Literal, Prediction, Rule


@dataclass(frozen=True)
class ProofCandidate:
    label: str
    support_ids: tuple[str, ...]
    depth: int
    strategy: str
    expansions: int


@dataclass(frozen=True)
class MPSSResult:
    label: str
    support_ids: tuple[str, ...]
    strategy: str
    expansions: int
    positive_found: bool
    negative_found: bool
    contradiction: bool


def mpss_predict(example: Example) -> Prediction:
    result = run_mpss(example.facts, example.rules, example.query)
    trace = {
        "strategy": result.strategy,
        "expansions": result.expansions,
        "positive_found": result.positive_found,
        "negative_found": result.negative_found,
        "contradiction": result.contradiction,
    }
    return Prediction(example.id, "mpss", result.label, result.support_ids, json.dumps(trace, sort_keys=True))


def mpss_predictions(examples: list[Example]) -> list[Prediction]:
    return [mpss_predict(example) for example in examples]


def run_mpss(facts: list[Fact], rules: list[Rule], query: Literal) -> MPSSResult:
    return run_mpss_with_plan(facts, rules, query, {})


def run_mpss_with_plan(facts: list[Fact], rules: list[Rule], query: Literal, plan: dict[str, Any] | None = None) -> MPSSResult:
    plan = plan or {}
    primary = str(plan.get("primary_strategy", "backward_chaining"))
    positive = _search_literal(facts, rules, query, "positive", primary)

    should_check_negation = bool(plan.get("check_negation", True))
    should_check_contradiction = bool(plan.get("check_contradiction", True))
    negative = _search_literal(facts, rules, query.negate(), "negative", primary) if should_check_negation or should_check_contradiction else None
    contradiction = positive is not None and negative is not None

    should_minimize = bool(plan.get("support_minimization", True))
    strategy_prefix = f"llm_{primary}" if plan else ""

    if contradiction:
        support_ids = tuple(sorted(set(positive.support_ids) | set(negative.support_ids)))
        support = _minimize_support(facts, rules, query, support_ids, "ambiguous") if should_minimize else support_ids
        strategy = "contradiction_check"
        if strategy_prefix:
            strategy = f"{strategy_prefix}_{strategy}"
        return MPSSResult("ambiguous", support, strategy, positive.expansions + negative.expansions, True, True, True)
    if positive is not None:
        support = _minimize_support(facts, rules, query, positive.support_ids, "true") if should_minimize else positive.support_ids
        strategy = positive.strategy
        if strategy_prefix:
            strategy = f"{strategy_prefix}_{strategy}"
        return MPSSResult("true", support, strategy, positive.expansions + (negative.expansions if negative else 0), True, False, False)
    if negative is not None:
        support = _minimize_support(facts, rules, query, negative.support_ids, "false") if should_minimize else negative.support_ids
        strategy = negative.strategy
        if strategy_prefix:
            strategy = f"{strategy_prefix}_{strategy}"
        return MPSSResult("false", support, strategy, negative.expansions, False, True, False)
    strategy = "stop_unknown"
    if strategy_prefix:
        strategy = f"{strategy_prefix}_{strategy}"
    return MPSSResult("unknown", (), strategy, 0, False, False, False)


def _search_literal(facts: list[Fact], rules: list[Rule], goal: Literal, polarity: str, primary_strategy: str = "backward_chaining") -> ProofCandidate | None:
    backward = _backward_prove(facts, rules, goal, seen=frozenset())
    forward_result = prove(facts, rules, goal)
    label = "true" if polarity == "positive" else "false"
    forward = None
    if forward_result.label == "true":
        forward = ProofCandidate(label, forward_result.support_ids, forward_result.depth, f"{polarity}_forward_fallback", len(forward_result.entailed))
    backward_candidate = ProofCandidate(label, backward[0], backward[1], f"{polarity}_backward_chain", backward[2]) if backward is not None else None

    if primary_strategy == "forward_expansion":
        if forward is not None:
            return forward
        return backward_candidate
    if primary_strategy == "negated_query_search" and polarity == "negative":
        if backward_candidate is not None:
            return backward_candidate
        return forward
    if backward_candidate is None:
        return forward
    if forward is not None and _candidate_key(forward) < _candidate_key(backward_candidate):
        return forward
    return backward_candidate


def _backward_prove(
    facts: list[Fact],
    rules: list[Rule],
    goal: Literal,
    seen: frozenset[Literal],
) -> tuple[tuple[str, ...], int, int] | None:
    if goal in seen:
        return None
    fact_matches = [fact for fact in facts if fact.literal == goal]
    if fact_matches:
        fact = sorted(fact_matches, key=lambda item: item.id)[0]
        return (fact.id,), 0, 1

    entities = {fact.literal.entity for fact in facts} | {goal.entity}
    best: tuple[tuple[str, ...], int, int] | None = None
    for rule in _rule_instances(rules, entities):
        if rule.consequent != goal:
            continue
        support = {rule.id}
        depth = 0
        expansions = 1
        failed = False
        for antecedent in rule.antecedents:
            antecedent_result = _backward_prove(facts, rules, antecedent, seen | {goal})
            expansions += antecedent_result[2] if antecedent_result else 1
            if antecedent_result is None:
                failed = True
                break
            support.update(antecedent_result[0])
            depth = max(depth, antecedent_result[1] + 1)
        if failed:
            continue
        candidate = (tuple(sorted(support)), depth, expansions)
        if best is None or (candidate[1], len(candidate[0]), candidate[0]) < (best[1], len(best[0]), best[0]):
            best = candidate
    return best


def _rule_instances(rules: list[Rule], entities: set[str]) -> list[Rule]:
    instances: list[Rule] = []
    for rule in rules:
        if rule.consequent.entity == "x" or any(antecedent.entity == "x" for antecedent in rule.antecedents):
            instances.extend(_instantiate_rule(rule, entity) for entity in sorted(entities))
        else:
            instances.append(rule)
    return instances


def _instantiate_rule(rule: Rule, entity: str) -> Rule:
    return Rule(
        rule.id,
        tuple(_instantiate_literal(antecedent, entity) for antecedent in rule.antecedents),
        _instantiate_literal(rule.consequent, entity),
    )


def _instantiate_literal(literal: Literal, entity: str) -> Literal:
    if literal.entity == "x":
        return Literal(literal.predicate, entity, literal.negated)
    return literal


def _minimize_support(facts: list[Fact], rules: list[Rule], query: Literal, support_ids: tuple[str, ...], target_label: str) -> tuple[str, ...]:
    current = set(support_ids)
    changed = True
    while changed:
        changed = False
        for premise_id in sorted(current):
            trial = current - {premise_id}
            result = prove(
                [fact for fact in facts if fact.id in trial],
                [rule for rule in rules if rule.id in trial],
                query,
            )
            if result.label == target_label:
                current = set(result.support_ids)
                changed = True
                break
    return tuple(sorted(current))


def _candidate_key(candidate: ProofCandidate) -> tuple[int, int, tuple[str, ...]]:
    return candidate.depth, len(candidate.support_ids), candidate.support_ids
