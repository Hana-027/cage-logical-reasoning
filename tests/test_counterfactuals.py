from cf_reasoning.counterfactuals import generate_counterfactuals
from cf_reasoning.generator import generate_examples, render_context
from cf_reasoning.prover import prove
from cf_reasoning.schema import Example, Fact, Literal, Rule


def test_counterfactuals_are_valid_and_linked():
    examples = generate_examples(20, seed=3, max_depth=3)
    cfs = generate_counterfactuals(examples, seed=3, max_per_example=4)
    assert cfs
    source_ids = {ex.id for ex in examples}
    assert all(cf.source_id in source_ids for cf in cfs)
    assert all(cf.label in {"true", "false", "unknown"} for cf in cfs)
    assert all(cf.changed_ids for cf in cfs)




def test_counterfactuals_include_changed_and_preserved():
    examples = generate_examples(50, seed=4, max_depth=3)
    cfs = generate_counterfactuals(examples, seed=4, max_per_example=4)
    relations = {cf.expected_relation for cf in cfs}
    assert "changed" in relations
    assert "preserved" in relations


def test_counterfactuals_include_proof_level_metadata():
    examples = generate_examples(50, seed=4, max_depth=3)
    cfs = generate_counterfactuals(examples, seed=4, max_per_example=9)
    families = {cf.cf_family for cf in cfs}
    assert "proof_breaking" in families
    assert "proof_preserving" in families
    assert "contradiction_injection" in families
    assert "distractor_injection" in families
    assert "entity_swap" in families
    assert "paraphrase_preserving" in families
    assert all(cf.bundle_id for cf in cfs)
    assert all(cf.parent_id for cf in cfs)
    assert all(cf.diagnostic_dimension for cf in cfs)
    assert all(cf.proof_relation for cf in cfs)
    assert all(cf.target_support_ids == cf.support_ids for cf in cfs)
    assert all(0.0 <= cf.support_overlap <= 1.0 for cf in cfs)


def test_support_shift_counterfactual_changes_gold_support_not_answer():
    facts = [Fact("F1", Literal("kind", "alice"))]
    rules = [
        Rule("R1", (Literal("kind", "alice"),), Literal("smart", "alice")),
        Rule("R2", (Literal("smart", "alice"),), Literal("blue", "alice")),
    ]
    query = Literal("blue", "alice")
    result = prove(facts, rules, query)
    source = Example("multi_hop", facts, rules, query, result.label, result.support_ids, result.depth, render_context(facts, rules, query))

    cfs = generate_counterfactuals([source], seed=1, max_per_example=9)
    support_shift = [cf for cf in cfs if cf.cf_family in {"support_shift", "alternate_proof"}]

    assert support_shift
    assert any(cf.expected_relation == "preserved" for cf in support_shift)
    assert any(set(cf.target_support_ids) != set(source.support_ids) for cf in support_shift)


def test_contradiction_and_paraphrase_diagnostics():
    facts = [Fact("F1", Literal("kind", "alice"))]
    rules = [Rule("R1", (Literal("kind", "alice"),), Literal("blue", "alice"))]
    query = Literal("blue", "alice")
    result = prove(facts, rules, query)
    source = Example("diagnostic", facts, rules, query, result.label, result.support_ids, result.depth, render_context(facts, rules, query))

    cfs = generate_counterfactuals([source], seed=1, max_per_example=9)
    contradiction = next(cf for cf in cfs if cf.cf_family == "contradiction_injection")
    paraphrase = next(cf for cf in cfs if cf.cf_family == "paraphrase_preserving")

    assert contradiction.expected_relation == "changed"
    assert contradiction.proof_relation == "conflicting_support"
    assert contradiction.conflict_label
    assert contradiction.is_minimal
    assert paraphrase.expected_relation == "preserved"
    assert paraphrase.text != source.text
    assert set(paraphrase.target_support_ids) == set(source.support_ids)
