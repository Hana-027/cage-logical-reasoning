from __future__ import annotations

import copy
import random
from collections import defaultdict

from .generator import render_context
from .prover import prove
from .schema import CounterfactualExample, Example, Fact, Literal, Rule


def _replace_fact(facts: list[Fact], fact_id: str, new_literal: Literal) -> list[Fact]:
    return [Fact(f.id, new_literal) if f.id == fact_id else f for f in facts]


def _replace_rule(rules: list[Rule], rule_id: str, new_rule: Rule) -> list[Rule]:
    return [new_rule if r.id == rule_id else r for r in rules]


def _delete_rule(rules: list[Rule], rule_id: str) -> list[Rule]:
    return [r for r in rules if r.id != rule_id]


def _delete_fact(facts: list[Fact], fact_id: str) -> list[Fact]:
    return [f for f in facts if f.id != fact_id]


def _support_jaccard(left: tuple[str, ...] | set[str], right: tuple[str, ...] | set[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set) if left_set | right_set else 0.0


def _proof_relation(
    source_label: str,
    target_label: str,
    source_support: tuple[str, ...],
    target_support: tuple[str, ...],
    conflict_label: str = "",
) -> str:
    if conflict_label:
        return "conflicting_support"
    if source_label != target_label:
        return "broken"
    source_set = set(source_support)
    target_set = set(target_support)
    if source_set == target_set:
        return "preserved_same_support"
    if source_set and target_set and source_set & target_set:
        return "shifted_support"
    return "preserved_new_support"


def _make_counterfactual(
    source: Example,
    cf_index: int,
    intervention_type: str,
    cf_family: str,
    changed_ids: tuple[str, ...],
    facts: list[Fact],
    rules: list[Rule],
    diagnostic_dimension: str = "",
    query: Literal | None = None,
    expected_relation_override: str = "",
    target_label_override: str = "",
    text_override: str = "",
    edit_distance: int = 0,
    is_minimal: bool = False,
    conflict_label: str = "",
) -> CounterfactualExample | None:
    target_query = query or source.query
    result = prove(facts, rules, target_query)
    if result.label == "ambiguous" and not target_label_override:
        return None
    target_label = target_label_override or result.label
    relation = expected_relation_override or ("changed" if target_label != source.label else "preserved")
    source_support = source.support_ids
    target_support = result.support_ids
    source_set = set(source_support)
    target_set = set(target_support)
    return CounterfactualExample(
        id=f"{source.id}_cf_{cf_index:02d}",
        source_id=source.id,
        intervention_type=intervention_type,
        changed_ids=changed_ids,
        facts=facts,
        rules=rules,
        query=target_query,
        label=target_label,
        support_ids=target_support,
        depth=result.depth,
        expected_relation=relation,
        text=text_override or render_context(facts, rules, target_query),
        split=source.split,
        bundle_id=f"{source.id}_bundle",
        parent_id=source.id,
        cf_family=cf_family,
        diagnostic_dimension=diagnostic_dimension,
        proof_relation=_proof_relation(source.label, target_label, source_support, target_support, conflict_label),
        source_support_ids=source_support,
        target_support_ids=target_support,
        removed_support_ids=tuple(sorted(source_set - target_set)),
        added_support_ids=tuple(sorted(target_set - source_set)),
        support_overlap=_support_jaccard(source_support, target_support),
        edit_distance=edit_distance,
        is_minimal=is_minimal,
        conflict_label=conflict_label,
    )


def _next_fact_id(facts: list[Fact]) -> str:
    used = {fact.id for fact in facts}
    index = 1
    while f"CF_F{index}" in used:
        index += 1
    return f"CF_F{index}"


def _next_rule_id(rules: list[Rule]) -> str:
    used = {rule.id for rule in rules}
    index = 1
    while f"CF_R{index}" in used:
        index += 1
    return f"CF_R{index}"


def _entities(source: Example) -> list[str]:
    names = {source.query.entity}
    for fact in source.facts:
        names.add(fact.literal.entity)
    for rule in source.rules:
        names.add(rule.consequent.entity)
        names.update(literal.entity for literal in rule.antecedents)
    return sorted(names)


def _swap_literal(literal: Literal, left: str, right: str) -> Literal:
    if literal.entity == left:
        return Literal(literal.predicate, right, literal.negated)
    if literal.entity == right:
        return Literal(literal.predicate, left, literal.negated)
    return literal


def _render_paraphrased_context(facts: list[Fact], rules: list[Rule], query: Literal) -> str:
    lines: list[str] = []
    for fact in facts:
        literal = fact.literal
        entity = literal.entity.capitalize()
        if literal.negated:
            lines.append(f"{fact.id}: It is false that {entity} is {literal.predicate}.")
        else:
            lines.append(f"{fact.id}: It is true that {entity} has the property of being {literal.predicate}.")
    for rule in rules:
        lhs = " and ".join(l.to_text().removesuffix(".").replace(l.entity.capitalize(), "someone") for l in rule.antecedents)
        rhs = rule.consequent.to_text().removesuffix(".").replace(rule.consequent.entity.capitalize(), "they")
        lines.append(f"{rule.id}: Whenever {lhs}, it follows that {rhs}.")
    lines.append(f"Query: Determine whether {query.to_text().removesuffix('.')}.")
    return "\n".join(lines)


def _support_shift_fact(source: Example, cf_index: int) -> CounterfactualExample | None:
    if source.label != "true" or source.query.negated:
        return None
    facts = copy.deepcopy(source.facts)
    rules = copy.deepcopy(source.rules)
    new_fact = Fact(_next_fact_id(facts), source.query)
    facts.append(new_fact)
    cf = _make_counterfactual(
        source,
        cf_index,
        "support_shift_add_redundant_fact",
        "support_shift",
        (new_fact.id,),
        facts,
        rules,
        diagnostic_dimension="support_shift_awareness",
        edit_distance=1,
    )
    if cf is None or cf.expected_relation != "preserved" or set(cf.target_support_ids) == set(source.support_ids):
        return None
    return cf


def _support_shift_bridge_rule(source: Example, cf_index: int) -> CounterfactualExample | None:
    if source.label != "true" or source.depth < 2:
        return None
    support_fact = next((fact for fact in source.facts if fact.id in set(source.support_ids)), None)
    if support_fact is None or support_fact.literal == source.query:
        return None
    facts = copy.deepcopy(source.facts)
    rules = copy.deepcopy(source.rules)
    new_rule = Rule(_next_rule_id(rules), (support_fact.literal,), source.query)
    rules.append(new_rule)
    cf = _make_counterfactual(
        source,
        cf_index,
        "support_shift_add_bridge_rule",
        "support_shift",
        (new_rule.id,),
        facts,
        rules,
        diagnostic_dimension="support_shift_awareness",
        edit_distance=1,
    )
    if cf is None or cf.expected_relation != "preserved" or set(cf.target_support_ids) == set(source.support_ids):
        return None
    return cf


def _alternate_proof_delete(source: Example, cf_index: int) -> CounterfactualExample | None:
    if source.label != "true" or not source.support_ids:
        return None
    facts = copy.deepcopy(source.facts)
    rules = copy.deepcopy(source.rules)
    new_fact = Fact(_next_fact_id(facts), source.query)
    facts.append(new_fact)
    support_rule = next((rule for rule in source.rules if rule.id in set(source.support_ids)), None)
    support_fact = next((fact for fact in source.facts if fact.id in set(source.support_ids)), None)
    removed_id = ""
    if support_rule is not None:
        rules = _delete_rule(rules, support_rule.id)
        removed_id = support_rule.id
    elif support_fact is not None:
        facts = _delete_fact(facts, support_fact.id)
        removed_id = support_fact.id
    if not removed_id:
        return None
    cf = _make_counterfactual(
        source,
        cf_index,
        "alternate_proof_support_delete",
        "alternate_proof",
        (new_fact.id, removed_id),
        facts,
        rules,
        diagnostic_dimension="support_shift_awareness",
        edit_distance=2,
    )
    if cf is None or cf.expected_relation != "preserved" or removed_id in set(cf.target_support_ids):
        return None
    return cf


def _contradiction_injection(source: Example, cf_index: int) -> CounterfactualExample | None:
    if source.label not in {"true", "false"}:
        return None
    facts = copy.deepcopy(source.facts)
    rules = copy.deepcopy(source.rules)
    contradictory_literal = source.query.negate() if source.label == "true" else source.query
    new_fact = Fact(_next_fact_id(facts), contradictory_literal)
    facts.append(new_fact)
    result = prove(facts, rules, source.query)
    target_label_override = "unknown" if result.label == "ambiguous" else ""
    cf = _make_counterfactual(
        source,
        cf_index,
        "inject_contradictory_fact",
        "contradiction_injection",
        (new_fact.id,),
        facts,
        rules,
        diagnostic_dimension="conflict_handling",
        expected_relation_override="changed",
        target_label_override=target_label_override,
        edit_distance=1,
        is_minimal=True,
        conflict_label="ambiguous" if result.label == "ambiguous" else "explicit_contradiction",
    )
    if cf is None:
        return None
    return cf


def _distractor_injection(source: Example, cf_index: int) -> CounterfactualExample | None:
    facts = copy.deepcopy(source.facts)
    rules = copy.deepcopy(source.rules)
    used_literals = {fact.literal for fact in facts}
    base_entity = f"cf_{source.query.entity}"
    entity = base_entity
    used_entities = set(_entities(source))
    index = 1
    while entity in used_entities:
        index += 1
        entity = f"{base_entity}_{index}"
    literal = Literal(source.query.predicate, entity, source.query.negated)
    if literal in used_literals:
        return None
    new_fact = Fact(_next_fact_id(facts), literal)
    facts.append(new_fact)
    cf = _make_counterfactual(
        source,
        cf_index,
        "lexical_distractor_injection",
        "distractor_injection",
        (new_fact.id,),
        facts,
        rules,
        diagnostic_dimension="distractor_robustness",
        edit_distance=1,
    )
    if cf is None or cf.expected_relation != "preserved" or set(cf.target_support_ids) != set(source.support_ids):
        return None
    return cf


def _entity_swap(source: Example, cf_index: int) -> CounterfactualExample | None:
    entities = _entities(source)
    left = source.query.entity
    right = next((entity for entity in entities if entity != left), "cf_entity")
    if left == right:
        return None
    facts = [Fact(fact.id, _swap_literal(fact.literal, left, right)) for fact in copy.deepcopy(source.facts)]
    rules = [
        Rule(
            rule.id,
            tuple(_swap_literal(literal, left, right) for literal in rule.antecedents),
            _swap_literal(rule.consequent, left, right),
        )
        for rule in copy.deepcopy(source.rules)
    ]
    query = _swap_literal(source.query, left, right)
    cf = _make_counterfactual(
        source,
        cf_index,
        "entity_name_swap",
        "entity_swap",
        (left, right),
        facts,
        rules,
        diagnostic_dimension="entity_binding",
        query=query,
    )
    if cf is None or cf.expected_relation != "preserved" or cf.text == source.text:
        return None
    return cf


def _rule_structure_intervention(source: Example, cf_index: int) -> CounterfactualExample | None:
    support = set(source.support_ids)
    rule = next((candidate for candidate in source.rules if candidate.id in support and candidate.antecedents), None)
    if rule is None:
        return None
    reversed_rule = Rule(rule.id, (rule.consequent,), rule.antecedents[0])
    facts = copy.deepcopy(source.facts)
    rules = _replace_rule(copy.deepcopy(source.rules), rule.id, reversed_rule)
    cf = _make_counterfactual(
        source,
        cf_index,
        "rule_direction_reverse",
        "rule_structure_intervention",
        (rule.id,),
        facts,
        rules,
        diagnostic_dimension="rule_understanding",
        edit_distance=1,
        is_minimal=True,
    )
    if cf is None or cf.expected_relation != "changed":
        return None
    return cf


def _paraphrase_preserving(source: Example, cf_index: int) -> CounterfactualExample | None:
    facts = copy.deepcopy(source.facts)
    rules = copy.deepcopy(source.rules)
    text = _render_paraphrased_context(facts, rules, source.query)
    if text == source.text:
        return None
    cf = _make_counterfactual(
        source,
        cf_index,
        "template_paraphrase",
        "paraphrase_preserving",
        ("TEXT",),
        facts,
        rules,
        diagnostic_dimension="semantic_invariance",
        text_override=text,
    )
    if cf is None or cf.expected_relation != "preserved" or set(cf.target_support_ids) != set(source.support_ids):
        return None
    return cf


def _choose_by_family(candidates: list[CounterfactualExample], max_per_example: int) -> list[CounterfactualExample]:
    by_family: dict[str, list[CounterfactualExample]] = defaultdict(list)
    for candidate in candidates:
        by_family[candidate.cf_family].append(candidate)
    chosen: list[CounterfactualExample] = []
    priority = (
        "proof_breaking",
        "proof_preserving",
        "support_shift",
        "alternate_proof",
        "contradiction_injection",
        "distractor_injection",
        "entity_swap",
        "rule_structure_intervention",
        "paraphrase_preserving",
    )
    for family in priority:
        if by_family[family] and len(chosen) < max_per_example:
            chosen.append(by_family[family][0])
    for candidate in candidates:
        if len(chosen) >= max_per_example:
            break
        if candidate not in chosen:
            chosen.append(candidate)
    return chosen[:max_per_example]


def generate_counterfactuals(
    examples: list[Example],
    seed: int = 42,
    max_per_example: int = 4,
) -> list[CounterfactualExample]:
    rng = random.Random(seed)
    counterfactuals: list[CounterfactualExample] = []

    for source in examples:
        support = set(source.support_ids)
        support_facts = [f for f in source.facts if f.id in support]
        support_rules = [r for r in source.rules if r.id in support]
        distractor_facts = [f for f in source.facts if f.id not in support]
        distractor_rules = [r for r in source.rules if r.id not in support]
        candidates: list[CounterfactualExample] = []

        for fact in support_facts[:2]:
            cf = _make_counterfactual(
                source,
                len(candidates) + 1,
                "support_fact_flip",
                "proof_breaking",
                (fact.id,),
                _replace_fact(copy.deepcopy(source.facts), fact.id, fact.literal.negate()),
                copy.deepcopy(source.rules),
                diagnostic_dimension="proof_causality",
                edit_distance=1,
                is_minimal=True,
            )
            if cf is not None:
                candidates.append(cf)

        for rule in support_rules[:2]:
            cf = _make_counterfactual(
                source,
                len(candidates) + 1,
                "support_rule_delete",
                "proof_breaking",
                (rule.id,),
                copy.deepcopy(source.facts),
                _delete_rule(copy.deepcopy(source.rules), rule.id),
                diagnostic_dimension="proof_causality",
                edit_distance=1,
                is_minimal=True,
            )
            if cf is not None:
                candidates.append(cf)

        rng.shuffle(distractor_facts)
        for fact in distractor_facts[:2]:
            cf = _make_counterfactual(
                source,
                len(candidates) + 1,
                "distractor_fact_flip",
                "proof_preserving",
                (fact.id,),
                _replace_fact(copy.deepcopy(source.facts), fact.id, fact.literal.negate()),
                copy.deepcopy(source.rules),
                diagnostic_dimension="proof_preservation",
                edit_distance=1,
                is_minimal=True,
            )
            if cf is not None:
                candidates.append(cf)

        rng.shuffle(distractor_rules)
        for rule in distractor_rules[:1]:
            cf = _make_counterfactual(
                source,
                len(candidates) + 1,
                "distractor_rule_delete",
                "proof_preserving",
                (rule.id,),
                copy.deepcopy(source.facts),
                _delete_rule(copy.deepcopy(source.rules), rule.id),
                diagnostic_dimension="proof_preservation",
                edit_distance=1,
                is_minimal=True,
            )
            if cf is not None:
                candidates.append(cf)

        for maker in (
            _support_shift_fact,
            _support_shift_bridge_rule,
            _alternate_proof_delete,
            _contradiction_injection,
            _distractor_injection,
            _entity_swap,
            _rule_structure_intervention,
            _paraphrase_preserving,
        ):
            cf = maker(source, len(candidates) + 1)
            if cf is not None:
                candidates.append(cf)

        counterfactuals.extend(_choose_by_family(candidates, max_per_example))

    return counterfactuals
