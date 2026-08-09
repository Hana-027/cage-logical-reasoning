from cf_reasoning.cage import cage_predict
from cf_reasoning.llm_client import _extract_label, _extract_premises
from cf_reasoning.schema import Example, Fact, Literal, Rule


def test_cage_runs_predict_diagnose_repair_loop():
    ex = Example(
        "ex",
        [Fact("F1", Literal("kind", "alice"))],
        [Rule("R1", (Literal("kind", "alice"),), Literal("blue", "alice"))],
        Literal("blue", "alice"),
        "true",
        ("F1", "R1"),
        1,
        "F1: Alice is kind.\nR1: If someone is kind, then they are blue.\nQuery: Alice is blue.",
    )
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if "Counterfactual problem" in prompt:
            return '{"answer":"unknown","brief_explanation":"The support was changed."}'
        if "repairing" in prompt:
            return '{"answer":"true","causal_premises":["F1","R1"],"brief_explanation":"F1 and R1 prove the query."}'
        return '{"answer":"true","causal_premises":["F1"],"brief_explanation":"Initial answer."}'

    pred, raw = cage_predict(ex, fake_call, _extract_label, _extract_premises)
    assert pred.method == "llm_cage"
    assert pred.label == "true"
    assert pred.premise_ids == ("F1", "R1")
    assert len(raw) == 2
    assert any("counterfactual causal attribution feedback" in call for call in calls)
