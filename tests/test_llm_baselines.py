import json

import pandas as pd
import pytest

from cf_reasoning.datasets.folio import FolioExample
from cf_reasoning.folio_cage import FolioCAGEConfig, folio_cage_predict, folio_cage_select_predict, folio_cpa_predict
from cf_reasoning.llm_baselines import folio_llm_baseline_rows, llm_baseline_predictions, normalize_baseline_methods
from cf_reasoning.llm_client import _canonicalize_results, _extract_folio_label, _extract_label, _extract_premises
from cf_reasoning.schema import Example, Fact, Literal, Rule


def _example() -> Example:
    return Example(
        "ex",
        [Fact("F1", Literal("kind", "alice"))],
        [Rule("R1", (Literal("kind", "alice"),), Literal("blue", "alice"))],
        Literal("blue", "alice"),
        "true",
        ("F1", "R1"),
        1,
        "F1: Alice is kind.\nR1: If someone is kind, then they are blue.\nQuery: Alice is blue.",
    )


def _folio_example() -> FolioExample:
    return FolioExample(
        "folio_x",
        ["All kind people are blue.", "Alice is kind."],
        ["∀x (Kind(x) -> Blue(x))", "Kind(alice)"],
        "Alice is blue.",
        "Blue(alice)",
        "true",
    )


def test_normalize_baseline_methods():
    assert normalize_baseline_methods("") == ()
    assert normalize_baseline_methods("all") == (
        "logiclm",
        "symbcot",
        "vericot",
        "direct_cage",
        "logiclm_cage",
        "symbcot_cage",
        "direct_cage_gated",
        "logiclm_cage_gated",
        "symbcot_cage_gated",
    )
    assert normalize_baseline_methods("direct_cage,logiclm_cage") == ("direct_cage", "logiclm_cage")
    assert normalize_baseline_methods("logiclm,vericot") == ("logiclm", "vericot")
    assert normalize_baseline_methods("logiclm,logiclm") == ("logiclm",)
    with pytest.raises(ValueError):
        normalize_baseline_methods("unknown")


def test_llm_baseline_predictions_with_fake_call_llm():
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if "LogicLM-style" in prompt:
            assert json_output is True
            assert schema is not None
            return '{"answer":"true","causal_premises":["F1","R1"],"brief_explanation":"Symbolic solver proves it."}'
        if "Symbolic Chain-of-Thought" in prompt:
            return '{"answer":"true","causal_premises":["F1"],"brief_explanation":"A symbolic proof sketch supports it."}'
        if "Task Description" in prompt or "parse the problem" in prompt:
            return "Predicates:\nKind(Alice, True)\nBlue(Alice, True)"
        if "derive a plan" in prompt or "generate a plan" in prompt:
            return "Use F1 and R1."
        if "execute each step" in prompt or "determine whether the value" in prompt:
            return "Apply Modus Ponens. Final answer: {true}"
        if "drafting an answer" in prompt:
            assert json_output is True
            assert schema is not None
            return '{"answer":"unknown","causal_premises":[],"brief_explanation":"Draft is unsure."}'
        assert json_output is True
        assert schema is not None
        return '{"verified":true,"answer":"true","causal_premises":["F1","R1"],"error_type":"unsupported_answer","brief_explanation":"Verifier repairs the draft."}'

    preds, raw = llm_baseline_predictions([_example()], ("logiclm", "symbcot", "vericot"), fake_call, _extract_label, _extract_premises)

    assert [pred.method for pred in preds] == ["llm_logiclm", "llm_symbcot", "llm_vericot"]
    assert [pred.label for pred in preds] == ["true", "true", "true"]
    assert preds[0].premise_ids == ("F1", "R1")
    assert {row["method"] for row in raw} == {"llm_logiclm_program", "llm_logiclm", "llm_symbcot_translation", "llm_symbcot_plan", "llm_symbcot", "llm_vericot_draft", "llm_vericot_verify"}
    assert any("parse the problem" in call or "Task Description" in call for call in calls)
    assert any("Task Description" in call for call in calls)
    assert any("derive a plan" in call or "generate a plan" in call for call in calls)
    assert any("execute each step" in call or "determine whether the value" in call for call in calls)


def test_llm_baseline_predictions_include_symbcot_cage_with_fake_call_llm():
    def fake_call(prompt, max_tokens, json_output, schema):
        if "Task Description" in prompt or "parse the problem" in prompt:
            return "Predicates:\nKind(Alice, True)"
        if "derive a plan" in prompt or "generate a plan" in prompt:
            return "Use the symbolic rule."
        if "execute each step" in prompt or "determine whether the value" in prompt:
            return "Final answer: {true}"
        if "counterfactual attribution probe" in prompt or "causally faithful" in prompt:
            return '{"answer":"unknown","brief_explanation":"Support changed."}'
        return '{"answer":"true","causal_premises":["F1","R1"],"brief_explanation":"Kept."}'

    preds, raw = llm_baseline_predictions([_example()], ("symbcot_cage",), fake_call, _extract_label, _extract_premises)

    assert [pred.method for pred in preds] == ["llm_symbcot_cage"]
    assert preds[0].label == "true"
    assert {row["method"] for row in raw} == {"llm_symbcot_translation", "llm_symbcot_plan", "llm_symbcot"}


def test_direct_and_logiclm_cage_keep_valid_base_labels_and_record_trace():
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if "fallback path of a Logic-LM-style" in prompt:
            assert json_output is True
            return '{"answer":"false","causal_premises":["F1"],"brief_explanation":"LogicLM says no."}'
        if "counterfactual attribution probe" in prompt or "causally faithful" in prompt:
            return '{"answer":"unknown","brief_explanation":"Probe disagrees."}'
        return "Final answer: true"

    preds, raw = llm_baseline_predictions([_example()], ("direct_cage", "logiclm_cage"), fake_call, _extract_label, _extract_premises)

    assert [pred.method for pred in preds] == ["llm_direct_cage", "llm_logiclm_cage"]
    assert [pred.label for pred in preds] == ["true", "false"]
    assert all(json.loads(pred.raw_response)["repair_triggered"] is False for pred in preds)
    assert all(json.loads(pred.raw_response)["diagnostics"] for pred in preds)
    assert {row["method"] for row in raw} == {"llm_direct", "llm_logiclm_program", "llm_logiclm"}
    assert not any("repairing a logical reasoning answer" in call for call in calls)


def test_method_cage_reuses_requested_base_predictions_for_pairwise_comparison():
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if "fallback path of a Logic-LM-style" in prompt:
            return '{"answer":"false","causal_premises":["F1"],"brief_explanation":"Base LogicLM answer."}'
        if "counterfactual attribution probe" in prompt or "causally faithful" in prompt:
            return '{"answer":"unknown","brief_explanation":"Probe disagrees."}'
        return "not executable program"

    preds, raw = llm_baseline_predictions([_example()], ("logiclm", "logiclm_cage"), fake_call, _extract_label, _extract_premises)
    base, caged = preds
    trace = json.loads(caged.raw_response)

    assert [pred.method for pred in preds] == ["llm_logiclm", "llm_logiclm_cage"]
    assert base.label == "false"
    assert caged.label == base.label
    assert trace["initial_answer"] == base.label
    assert trace["final_answer"] == base.label
    assert trace["repair_triggered"] is False
    assert [row["method"] for row in raw] == ["llm_logiclm_program", "llm_logiclm"]
    assert sum("fallback path of a Logic-LM-style" in call for call in calls) == 1


def test_symbcot_cage_uses_conservative_wrapper_for_valid_base_label():
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if "Task Description" in prompt or "parse the problem" in prompt:
            return "Predicates:\nKind(Alice, True)"
        if "derive a plan" in prompt or "generate a plan" in prompt:
            return "Use the symbolic rule."
        if "execute each step" in prompt or "determine whether the value" in prompt:
            return "Final answer: {false}"
        if "counterfactual attribution probe" in prompt or "causally faithful" in prompt:
            return '{"answer":"true","brief_explanation":"Probe strongly disagrees."}'
        if "repairing a logical reasoning answer" in prompt:
            raise AssertionError("Repair should not run for a valid SymbCoT label")
        raise AssertionError(prompt[:120])

    preds, raw = llm_baseline_predictions([_example()], ("symbcot", "symbcot_cage"), fake_call, _extract_label, _extract_premises)
    base, caged = preds
    trace = json.loads(caged.raw_response)

    assert [pred.method for pred in preds] == ["llm_symbcot", "llm_symbcot_cage"]
    assert base.label == "false"
    assert caged.label == base.label
    assert trace["base_method"] == "llm_symbcot"
    assert trace["repair_triggered"] is False
    assert trace["final_answer"] == base.label
    assert {row["method"] for row in raw} == {"llm_symbcot_translation", "llm_symbcot_plan", "llm_symbcot"}


def test_method_cage_repairs_only_invalid_initial_labels():
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if "counterfactual attribution probe" in prompt or "causally faithful" in prompt:
            return '{"answer":"unknown","brief_explanation":"Probe can still run."}'
        if "repairing a logical reasoning answer" in prompt:
            return '{"answer":"unknown","causal_premises":["F1"],"brief_explanation":"Repair valid."}'
        return "Final answer: maybe"

    preds, raw = llm_baseline_predictions([_example()], ("direct_cage",), fake_call, _extract_label, _extract_premises)
    trace = json.loads(preds[0].raw_response)

    assert preds[0].method == "llm_direct_cage"
    assert preds[0].label == "unknown"
    assert preds[0].premise_ids == ("F1",)
    assert trace["initial_answer"] == "invalid"
    assert trace["repair_triggered"] is True
    assert trace["repair_reason"] == "invalid_initial_label"
    assert trace["final_answer"] == "unknown"
    assert [row["method"] for row in raw] == ["llm_direct", "llm_direct_cage"]


def test_gated_cage_repairs_only_with_strong_diagnostic_majority():
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if "counterfactual attribution probe" in prompt or "causally faithful" in prompt:
            return '{"answer":"false","brief_explanation":"The probe contradicts the base answer."}'
        if "repairing a logical reasoning answer" in prompt:
            return '{"answer":"false","causal_premises":["F1"],"brief_explanation":"The diagnostic majority supports false."}'
        return "Final answer: true"

    preds, raw = llm_baseline_predictions([_example()], ("direct_cage_gated",), fake_call, _extract_label, _extract_premises)
    trace = json.loads(preds[0].raw_response)

    assert preds[0].method == "llm_direct_cage_gated"
    assert preds[0].label == "false"
    assert trace["repair_triggered"] is True
    assert trace["repair_accepted"] is True
    assert trace["gated_target_answer"] == "false"
    assert raw[-1]["method"] == "llm_direct_cage_gated"


def test_gated_cage_keeps_valid_base_when_diagnostics_are_mixed():
    probe_calls = 0

    def fake_call(prompt, max_tokens, json_output, schema):
        nonlocal probe_calls
        if "counterfactual attribution probe" in prompt or "causally faithful" in prompt:
            probe_calls += 1
            answer = "false" if probe_calls == 2 else "true"
            return f'{{"answer":"{answer}","brief_explanation":"A mixed probe result."}}'
        if "repairing a logical reasoning answer" in prompt:
            raise AssertionError("Mixed diagnostics must not trigger gated repair")
        return "Final answer: true"

    preds, raw = llm_baseline_predictions([_example()], ("direct_cage_gated",), fake_call, _extract_label, _extract_premises)
    trace = json.loads(preds[0].raw_response)

    assert preds[0].method == "llm_direct_cage_gated"
    assert preds[0].label == "true"
    assert trace["repair_triggered"] is False
    assert trace["repair_accepted"] is False
    assert trace["gated_target_answer"] is None
    assert raw == [{"example_id": "ex", "method": "llm_direct", "raw_response": "Final answer: true"}]



    def fake_call(prompt, max_tokens, json_output, schema):
        if "drafting an answer" in prompt:
            return '{"answer":"false","causal_premises":["F1"],"brief_explanation":"Bad draft."}'
        return '{"verified":false,"answer":"true","causal_premises":["F1","R1"],"error_type":"invalid_derivation","brief_explanation":"R1 proves the query."}'

    preds, raw = llm_baseline_predictions([_example()], ("vericot",), fake_call, _extract_label, _extract_premises)

    assert len(preds) == 1
    assert preds[0].method == "llm_vericot"
    assert preds[0].label == "true"
    assert preds[0].premise_ids == ("F1", "R1")
    trace = json.loads(preds[0].raw_response)
    assert trace["draft_answer"] == "false"
    assert trace["final_answer"] == "true"
    assert [row["method"] for row in raw] == ["llm_vericot_draft", "llm_vericot_verify"]


def test_vericot_keeps_draft_when_verifier_marks_valid_without_answer():
    def fake_call(prompt, max_tokens, json_output, schema):
        if "drafting an answer" in prompt:
            return '{"answer":"No","causal_premises":["F1","R1"],"brief_explanation":"Draft proves negation."}'
        return '{"is_valid":true,"revised_draft":null}'

    preds, raw = llm_baseline_predictions([_example()], ("vericot",), fake_call, _extract_label, _extract_premises)

    assert preds[0].method == "llm_vericot"
    assert preds[0].label == "false"
    assert preds[0].premise_ids == ("F1", "R1")
    assert [row["method"] for row in raw] == ["llm_vericot_draft", "llm_vericot_verify"]


def test_folio_llm_baseline_rows_with_fake_call_llm():
    def fake_call(prompt, max_tokens, json_output, schema):
        if "LogicLM-style" in prompt:
            assert "Formal representation" in prompt
            return '{"answer":"true","brief_explanation":"The FOL formula entails the conclusion."}'
        if "Symbolic Chain-of-Thought" in prompt:
            return '{"answer":"entailment","brief_explanation":"The symbolic proof succeeds."}'
        if "parse the problem" in prompt or "first-order logic formular" in prompt:
            return "Premises:\n∀x Kind(x) -> Blue(x)\nConclusion: Blue(alice)"
        if "generate a plan" in prompt or "derive a step by step plan" in prompt:
            return "Instantiate the rule."
        if "determine whether the value" in prompt or "execute each step" in prompt:
            return "Final answer: {entailment}"
        if "drafting an answer" in prompt:
            return '{"answer":"unknown","brief_explanation":"Draft is unsure."}'
        return '{"verified":false,"answer":"true","error_type":"unsupported_answer","brief_explanation":"Verifier repairs the draft."}'

    rows, raw = folio_llm_baseline_rows([_folio_example()], ("logiclm", "symbcot", "vericot"), fake_call, _extract_folio_label)

    assert [row["method"] for row in rows] == ["llm_logiclm", "llm_symbcot", "llm_vericot"]
    assert [row["pred"] for row in rows] == ["true", "true", "true"]
    assert [row["accuracy"] for row in rows] == [1, 1, 1]
    assert {row["method"] for row in raw} == {"llm_logiclm_program", "llm_logiclm", "llm_symbcot_translation", "llm_symbcot_plan", "llm_symbcot", "llm_vericot_draft", "llm_vericot_verify"}


def test_folio_transfer_cage_methods_with_fake_call_llm():
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if "diverse candidate answers" in prompt:
            return '{"candidates":[{"answer":"unknown","causal_premises":[],"brief_explanation":"unsure"},{"answer":"true","causal_premises":["P1","P2"],"brief_explanation":"supported"}]}'
        if "Repair" in prompt or "revising" in prompt:
            return '{"answer":"true","causal_premises":["P1","P2"],"brief_explanation":"Conservative repair keeps the supported label."}'
        if "using only" in prompt or "negated-conclusion" in prompt or "Verify" in prompt:
            return '{"answer":"true","agreement":"agree","brief_explanation":"The view agrees."}'
        return '{"answer":"true","causal_premises":["P1","P2"],"brief_explanation":"The premises entail the conclusion."}'

    ex = _folio_example()
    preds_and_raw = [
        folio_cpa_predict(ex, fake_call, _extract_folio_label, FolioCAGEConfig(max_tokens=128)),
        folio_cage_predict(ex, fake_call, _extract_folio_label, FolioCAGEConfig(max_tokens=128)),
        folio_cage_select_predict(ex, fake_call, _extract_folio_label, FolioCAGEConfig(max_tokens=128, n_candidates=2)),
    ]
    preds = [pred for pred, _ in preds_and_raw]

    assert [pred.method for pred in preds] == ["llm_cpa", "llm_cage", "llm_cage_select"]
    assert [pred.label for pred in preds] == ["true", "true", "true"]
    assert all("P1" in pred.premise_ids for pred in preds)
    assert any("natural-language premises and formal representation" in call for call in calls)
    assert any("not gold support annotations" in call for call in calls)


def test_folio_method_cage_wrappers_use_transfer_diagnostics_and_preserve_valid_labels():
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if "LogicLM-style" in prompt:
            return '{"answer":"true","brief_explanation":"The FOL formula entails the conclusion."}'
        if "parse the problem" in prompt or "first-order logic formular" in prompt:
            return "Premises:\n∀x Kind(x) -> Blue(x)\nConclusion: Blue(alice)"
        if "generate a plan" in prompt or "derive a step by step plan" in prompt:
            return "Instantiate the rule."
        if "determine whether the value" in prompt or "execute each step" in prompt:
            return "Final answer: {contradiction}"
        if "using only" in prompt or "negated-conclusion" in prompt or "Verify" in prompt:
            return '{"answer":"true","agreement":"agree","brief_explanation":"The transfer view disagrees with SymbCoT."}'
        if "Repair a FOLIO answer" in prompt:
            raise AssertionError("Repair should not run for valid FOLIO wrapper labels")
        return "Final answer: true"

    rows, raw = folio_llm_baseline_rows([_folio_example()], ("direct_cage", "logiclm_cage", "symbcot", "symbcot_cage"), fake_call, _extract_folio_label)

    assert [row["method"] for row in rows] == ["llm_direct_cage", "llm_logiclm_cage", "llm_symbcot", "llm_symbcot_cage"]
    assert [row["pred"] for row in rows] == ["true", "true", "false", "false"]
    assert {row["method"] for row in raw} == {"llm_direct", "llm_logiclm_program", "llm_logiclm", "llm_symbcot_translation", "llm_symbcot_plan", "llm_symbcot"}
    assert any("negated-conclusion" in call for call in calls)


def test_folio_method_cage_repairs_invalid_initial_label():
    def fake_call(prompt, max_tokens, json_output, schema):
        if "using only" in prompt or "negated-conclusion" in prompt or "Verify" in prompt:
            return '{"answer":"unknown","agreement":"uncertain","brief_explanation":"Probe is uncertain."}'
        if "Repair a FOLIO answer" in prompt:
            return '{"answer":"false","causal_premises":["P1"],"brief_explanation":"Repair valid."}'
        return "Final answer: maybe"

    rows, raw = folio_llm_baseline_rows([_folio_example()], ("direct_cage",), fake_call, _extract_folio_label)

    assert rows[0]["method"] == "llm_direct_cage"
    assert rows[0]["pred"] == "false"
    assert [row["method"] for row in raw] == ["llm_direct", "llm_direct_cage"]


def test_canonicalize_results_normalizes_folio_method_aliases():
    df = pd.DataFrame(
        [
            {"split": "folio", "example_id": "folio_1", "method": "folio_llm_logiclm", "gold": "true", "pred": "true", "accuracy": 1},
            {"split": "folio", "example_id": "folio_1", "method": "folio_llm_direct_cage", "gold": "true", "pred": "true", "accuracy": 1},
            {"split": "folio", "example_id": "folio_1", "method": "folio_llm_logiclm_cage", "gold": "true", "pred": "false", "accuracy": 0},
            {"split": "folio", "example_id": "folio_1", "method": "folio_llm_cage_select", "gold": "true", "pred": "unknown", "accuracy": 0},
        ]
    )

    out = _canonicalize_results(df, "folio")

    assert list(out.columns[:7]) == ["benchmark", "split", "method", "example_id", "gold", "pred", "accuracy"]
    assert set(out["method"]) == {"llm_logiclm", "llm_direct_cage", "llm_logiclm_cage", "llm_cage_select"}
    assert set(out["benchmark"]) == {"folio"}
