import json

from cf_reasoning.llm_mpss import fallback_plan, llm_guided_mpss_predict
from cf_reasoning.mpss import run_mpss_with_plan
from cf_reasoning.schema import Example, Fact, Literal, Rule


def _example():
    return Example(
        "ex",
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
    )


def test_run_mpss_with_plan_uses_strategy_prefix():
    ex = _example()
    plan = fallback_plan() | {"primary_strategy": "forward_expansion"}
    result = run_mpss_with_plan(ex.facts, ex.rules, ex.query, plan)
    assert result.label == "true"
    assert result.strategy.startswith("llm_forward_expansion")


def test_llm_guided_mpss_prediction_records_plan_trace():
    ex = _example()
    plan = fallback_plan()
    pred = llm_guided_mpss_predict(ex, plan, raw_plan='{"primary_strategy":"backward_chaining"}')
    trace = json.loads(pred.raw_response)
    assert pred.method == "llm_guided_mpss"
    assert pred.label == "true"
    assert trace["llm_plan"]["primary_strategy"] == "backward_chaining"
    assert trace["positive_found"] is True
