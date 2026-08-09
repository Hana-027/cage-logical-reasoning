import json

from cf_reasoning.evaluate import f1_score, summarize_base, summarize_counterfactual, support_jaccard
from cf_reasoning.generator import generate_examples, write_jsonl
from cf_reasoning.schema import CounterfactualExample, Example, Fact, Literal, Prediction


def test_f1_score_partial_overlap():
    p, r, f1 = f1_score({"F1", "R1"}, {"F1", "R2"})
    assert p == 0.5
    assert r == 0.5
    assert f1 == 0.5


def test_write_jsonl_accepts_dict_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"example_id": "ex_1", "raw_response": "true"}])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"example_id": "ex_1", "raw_response": "true"}]




def test_summarize_base_accuracy():
    examples = generate_examples(5, seed=2, max_depth=2)
    preds = [Prediction(ex.id, "test", ex.label, ex.support_ids) for ex in examples]
    df = summarize_base(examples, preds)
    assert df["accuracy"].mean() == 1.0
    assert df["attr_f1"].mean() == 1.0


def test_support_jaccard():
    assert support_jaccard({"F1", "R1"}, {"F1", "R2"}) == 1 / 3
    assert support_jaccard(set(), set()) == 1.0


def test_summarize_counterfactual_proof_level_metrics():
    base = Example(
        "ex1",
        [Fact("F1", Literal("kind", "alice"))],
        [],
        Literal("blue", "alice"),
        "true",
        ("F1", "R1"),
        1,
        "",
        split="test",
    )
    cf = CounterfactualExample(
        "cf1",
        "ex1",
        "support_shift_add_redundant_fact",
        ("CF_F1",),
        [Fact("CF_F1", Literal("blue", "alice"))],
        [],
        Literal("blue", "alice"),
        "true",
        ("CF_F1",),
        0,
        "preserved",
        "",
        split="test",
        bundle_id="ex1_bundle",
        parent_id="ex1",
        cf_family="support_shift",
        diagnostic_dimension="support_shift_awareness",
        proof_relation="preserved_new_support",
        source_support_ids=("F1", "R1"),
        target_support_ids=("CF_F1",),
        removed_support_ids=("F1", "R1"),
        added_support_ids=("CF_F1",),
        support_overlap=0.0,
    )
    pred = Prediction("cf1", "test", "true", ("CF_F1",))

    df = summarize_counterfactual([base], [cf], [pred])

    assert df.iloc[0]["cf_family"] == "support_shift"
    assert df.iloc[0]["diagnostic_dimension"] == "support_shift_awareness"
    assert df.iloc[0]["bundle_id"] == "ex1_bundle"
    assert df.iloc[0]["pred_target_support_jaccard"] == 1.0
    assert df.iloc[0]["support_shift_detected"] == 1
    assert df.iloc[0]["attribution_consistency"] == 1
    assert df.iloc[0]["alternate_proof_awareness"] == 1


def test_summarize_counterfactual_diagnostic_metrics():
    base = Example(
        "ex1",
        [Fact("F1", Literal("blue", "alice"))],
        [],
        Literal("blue", "alice"),
        "true",
        ("F1",),
        0,
        "",
        split="test",
    )
    cf = CounterfactualExample(
        "cf1",
        "ex1",
        "inject_contradictory_fact",
        ("CF_F1",),
        [Fact("F1", Literal("blue", "alice")), Fact("CF_F1", Literal("blue", "alice", True))],
        [],
        Literal("blue", "alice"),
        "unknown",
        ("CF_F1", "F1"),
        0,
        "changed",
        "",
        split="test",
        bundle_id="ex1_bundle",
        parent_id="ex1",
        cf_family="contradiction_injection",
        diagnostic_dimension="conflict_handling",
        proof_relation="conflicting_support",
        source_support_ids=("F1",),
        target_support_ids=("CF_F1", "F1"),
        added_support_ids=("CF_F1",),
        support_overlap=0.5,
        edit_distance=1,
        is_minimal=True,
        conflict_label="ambiguous",
    )
    pred = Prediction("cf1", "test", "unknown", ())

    df = summarize_counterfactual([base], [cf], [pred])

    assert df.iloc[0]["contradiction_detection"] == 1
    assert df.iloc[0]["minimal_edit_sensitivity"] == 1
    assert df.iloc[0]["conflict_label"] == "ambiguous"
