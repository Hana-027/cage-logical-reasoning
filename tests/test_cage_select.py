from cf_reasoning.cage_select import cage_select_predict
from cf_reasoning.llm_client import _extract_label, _extract_premises
from cf_reasoning.schema import Example, Fact, Literal, Rule


def test_cage_select_generates_scores_and_repairs_candidate():
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
        if "Generate 3 diverse candidate" in prompt:
            return '{"candidates":[{"answer":"unknown","causal_premises":[],"brief_explanation":"not sure"},{"answer":"true","causal_premises":["F1","R1"],"brief_explanation":"proof"}]}'
        if "Counterfactual problem" in prompt:
            return '{"answer":"unknown","brief_explanation":"changed support"}'
        if "revising the best candidate" in prompt:
            return '{"answer":"true","causal_premises":["F1","R1"],"brief_explanation":"F1 and R1 prove it."}'
        return '{"answer":"unknown"}'

    pred, raw = cage_select_predict(ex, fake_call, _extract_label, _extract_premises)
    assert pred.method == "llm_cage_select"
    assert pred.label == "true"
    assert pred.premise_ids == ("F1", "R1")
    assert len(raw) == 2
    assert any("Counterfactual problem" in call for call in calls)
    assert any("counterfactual stability score" in call for call in calls)
