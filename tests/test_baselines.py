from cf_reasoning.baselines import fact_lookup_predictions, one_step_rule_predictions, train_majority_predictions
from cf_reasoning.evaluate import aggregate_research, summarize_base, summarize_counterfactual
from cf_reasoning.schema import CounterfactualExample, Example, Fact, Literal, Rule


def _example(example_id: str, label: str, split: str = "dev") -> Example:
    facts = [Fact("F1", Literal("blue", "alice"))]
    return Example(example_id, facts, [], Literal("blue", "alice"), label, ("F1",), 0, "F1: Alice is blue.", split=split)


def test_train_majority_uses_train_distribution():
    train = [_example("tr1", "false", "train"), _example("tr2", "false", "train"), _example("tr3", "true", "train")]
    dev = [_example("dev1", "true", "dev")]
    preds = train_majority_predictions(dev, train)
    assert preds[0].label == "false"
    assert preds[0].method == "train_majority"


def test_fact_lookup_predicts_explicit_fact_and_negation():
    examples = [
        _example("ex1", "true"),
        Example("ex2", [Fact("F1", Literal("blue", "alice", True))], [], Literal("blue", "alice"), "false", ("F1",), 0, "", split="dev"),
    ]
    preds = fact_lookup_predictions(examples)
    assert [pred.label for pred in preds] == ["true", "false"]
    assert preds[0].premise_ids == ("F1",)


def test_one_step_rule_does_not_chain_multiple_hops():
    examples = [
        Example(
            "ex1",
            [Fact("F1", Literal("kind", "alice"))],
            [
                Rule("R1", (Literal("kind", "alice"),), Literal("smart", "alice")),
                Rule("R2", (Literal("smart", "alice"),), Literal("blue", "alice")),
            ],
            Literal("blue", "alice"),
            "true",
            ("F1", "R1", "R2"),
            2,
            "",
            split="dev",
        )
    ]
    preds = one_step_rule_predictions(examples)
    assert preds[0].label == "unknown"




def test_aggregate_research_includes_split_and_ci_columns():
    examples = [_example("ex1", "true", "dev")]
    preds = fact_lookup_predictions(examples)
    base_df = summarize_base(examples, preds)
    summary = aggregate_research(base_df, base_df.iloc[0:0])
    assert set(["scope", "split", "method", "accuracy", "accuracy_ci_low", "accuracy_ci_high"]).issubset(summary.columns)
    assert summary.iloc[0]["split"] == "dev"


def test_aggregate_research_supports_relation_grouping():
    base = _example("ex1", "true", "test")
    cf = CounterfactualExample(
        "cf1",
        "ex1",
        "irrelevant_fact",
        ("F2",),
        base.facts,
        base.rules,
        base.query,
        "true",
        base.support_ids,
        0,
        "preserved",
        base.text,
        split="test",
        bundle_id="ex1_bundle",
        parent_id="ex1",
        cf_family="proof_preserving",
        diagnostic_dimension="proof_preservation",
        proof_relation="preserved_same_support",
        source_support_ids=base.support_ids,
        target_support_ids=base.support_ids,
        support_overlap=1.0,
    )
    cf_df = summarize_counterfactual([base], [cf], fact_lookup_predictions([cf]))
    summary = aggregate_research(summarize_base([base], fact_lookup_predictions([base])), cf_df, group_cols=["split", "method", "expected_relation"])
    assert "expected_relation" in summary.columns
    assert summary[summary["scope"] == "counterfactual"].iloc[0]["expected_relation"] == "preserved"

    family_summary = aggregate_research(summarize_base([base], fact_lookup_predictions([base])), cf_df, group_cols=["split", "method", "cf_family"])
    proof_summary = aggregate_research(summarize_base([base], fact_lookup_predictions([base])), cf_df, group_cols=["split", "method", "proof_relation"])
    dimension_summary = aggregate_research(summarize_base([base], fact_lookup_predictions([base])), cf_df, group_cols=["split", "method", "diagnostic_dimension"])
    assert family_summary[family_summary["scope"] == "counterfactual"].iloc[0]["cf_family"] == "proof_preserving"
    assert proof_summary[proof_summary["scope"] == "counterfactual"].iloc[0]["proof_relation"] == "preserved_same_support"
    assert dimension_summary[dimension_summary["scope"] == "counterfactual"].iloc[0]["diagnostic_dimension"] == "proof_preservation"
