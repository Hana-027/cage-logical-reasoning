from cf_reasoning.prover import prove
from cf_reasoning.schema import Fact, Literal, Rule


def test_forward_chain_true():
    facts = [Fact("F1", Literal("kind", "alice"))]
    rules = [Rule("R1", (Literal("kind", "alice"),), Literal("blue", "alice"))]
    result = prove(facts, rules, Literal("blue", "alice"))
    assert result.label == "true"
    assert set(result.support_ids) == {"F1", "R1"}
    assert result.depth == 1


def test_false_from_negated_literal():
    facts = [Fact("F1", Literal("blue", "alice", True))]
    result = prove(facts, [], Literal("blue", "alice"))
    assert result.label == "false"
    assert result.support_ids == ("F1",)


def test_unknown_when_not_entailed():
    facts = [Fact("F1", Literal("kind", "alice"))]
    result = prove(facts, [], Literal("blue", "alice"))
    assert result.label == "unknown"
    assert result.support_ids == ()


def test_ambiguous_when_both_entailed():
    facts = [Fact("F1", Literal("blue", "alice")), Fact("F2", Literal("blue", "alice", True))]
    result = prove(facts, [], Literal("blue", "alice"))
    assert result.label == "ambiguous"
