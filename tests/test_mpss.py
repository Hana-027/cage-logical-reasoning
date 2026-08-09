import json

from cf_reasoning.mpss import mpss_predict
from cf_reasoning.schema import Example, Fact, Literal, Rule


def _predict(facts, rules, query, label="unknown", support_ids=()):
    ex = Example("ex", facts, rules, query, label, support_ids, 0, "", split="test")
    return mpss_predict(ex)


def test_mpss_proves_multihop_true_where_one_step_fails():
    pred = _predict(
        [Fact("F1", Literal("kind", "alice"))],
        [
            Rule("R1", (Literal("kind", "alice"),), Literal("smart", "alice")),
            Rule("R2", (Literal("smart", "alice"),), Literal("blue", "alice")),
        ],
        Literal("blue", "alice"),
        "true",
        ("F1", "R1", "R2"),
    )

    trace = json.loads(pred.raw_response)
    assert pred.label == "true"
    assert set(pred.premise_ids) == {"F1", "R1", "R2"}
    assert trace["strategy"] == "positive_backward_chain"
    assert trace["positive_found"] is True


def test_mpss_returns_false_when_negated_query_is_provable():
    pred = _predict(
        [Fact("F1", Literal("kind", "alice"))],
        [Rule("R1", (Literal("kind", "alice"),), Literal("blue", "alice", True))],
        Literal("blue", "alice"),
        "false",
        ("F1", "R1"),
    )

    trace = json.loads(pred.raw_response)
    assert pred.label == "false"
    assert set(pred.premise_ids) == {"F1", "R1"}
    assert trace["negative_found"] is True
    assert trace["positive_found"] is False


def test_mpss_returns_unknown_when_neither_query_nor_negation_is_provable():
    pred = _predict([Fact("F1", Literal("kind", "alice"))], [], Literal("blue", "alice"))

    trace = json.loads(pred.raw_response)
    assert pred.label == "unknown"
    assert pred.premise_ids == ()
    assert trace["strategy"] == "stop_unknown"


def test_mpss_maps_conflicting_proofs_to_ambiguous():
    pred = _predict(
        [Fact("F1", Literal("blue", "alice")), Fact("F2", Literal("blue", "alice", True))],
        [],
        Literal("blue", "alice"),
        "ambiguous",
        ("F1", "F2"),
    )

    trace = json.loads(pred.raw_response)
    assert pred.label == "ambiguous"
    assert set(pred.premise_ids) == {"F1", "F2"}
    assert trace["contradiction"] is True


def test_mpss_minimizes_support_and_ignores_distractors():
    pred = _predict(
        [
            Fact("F1", Literal("kind", "alice")),
            Fact("F2", Literal("furry", "alice")),
            Fact("F3", Literal("round", "alice")),
        ],
        [
            Rule("R1", (Literal("kind", "alice"),), Literal("smart", "alice")),
            Rule("R2", (Literal("smart", "alice"),), Literal("blue", "alice")),
            Rule("R3", (Literal("furry", "alice"), Literal("round", "alice")), Literal("red", "alice")),
        ],
        Literal("blue", "alice"),
        "true",
        ("F1", "R1", "R2"),
    )

    assert pred.label == "true"
    assert set(pred.premise_ids) == {"F1", "R1", "R2"}
    assert "F2" not in pred.premise_ids
    assert "F3" not in pred.premise_ids
