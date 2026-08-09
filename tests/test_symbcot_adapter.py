from pathlib import Path

from cf_reasoning.datasets.folio import FolioExample
from cf_reasoning.llm_client import _extract_folio_label, _extract_label, _extract_premises
from cf_reasoning.schema import Example, Fact, Literal, Rule
from cf_reasoning.symbcot_adapter import SymbCoTConfig, folio_symbcot_cage_predict, folio_symbcot_predict, symbcot_cage_predict, symbcot_predict


def _prompt_root(tmp_path: Path) -> Path:
    for dataset in ["ProofWriter", "FOLIO"]:
        folder = tmp_path / dataset
        folder.mkdir(parents=True)
        (folder / "translation.txt").write_text("Translate [[CONTEXT]] / [[QUESTION]]", encoding="utf-8")
        (folder / "plan_generation.txt").write_text("Plan [[CONTEXT]]", encoding="utf-8")
        (folder / "solver.txt").write_text("Solve [[CONTEXT]] with [[PLAN]]", encoding="utf-8")
    return tmp_path


def _example() -> Example:
    return Example(
        "ex",
        [Fact("F1", Literal("kind", "alice"))],
        [Rule("R1", (Literal("kind", "alice"),), Literal("blue", "alice"))],
        Literal("blue", "alice"),
        "true",
        ("F1", "R1"),
        1,
        "F1: Alice is kind.\nR1: If someone is kind, then they are blue.\nQuery: Is it true that Alice is blue?",
        split="proofwriter",
    )


def test_symbcot_predict_runs_translation_plan_solver(tmp_path):
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append((prompt, json_output, schema))
        if prompt.startswith("Translate"):
            return "Predicates:\nKind(alice)\nConclusion: Blue(alice)"
        if prompt.startswith("Plan"):
            return "Use F1 and R1."
        return "Step 1: Apply Modus Ponens. Final answer: {true}"

    pred, raw = symbcot_predict(_example(), fake_call, _extract_label, _extract_premises, SymbCoTConfig(max_tokens=128, prompt_root=_prompt_root(tmp_path)))

    assert pred.method == "llm_symbcot"
    assert pred.label == "true"
    assert [row["method"] for row in raw] == ["llm_symbcot_translation", "llm_symbcot_plan", "llm_symbcot"]
    assert all(json_output is False for _, json_output, _ in calls)
    assert calls[0][0] == "Translate F1: Alice is kind.\nR1: If someone is kind, then they are blue. / Is it true that Alice is blue?"



def test_symbcot_cage_keeps_valid_symbcot_answer_while_recording_probes(tmp_path):
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if prompt.startswith("Translate"):
            return "Predicates:\nKind(alice)\nConclusion: Blue(alice)"
        if prompt.startswith("Plan"):
            return "Use F1 and R1."
        if prompt.startswith("Solve"):
            return "Step 1: Apply Modus Ponens. Final answer: {true}"
        if "counterfactual attribution probe" in prompt:
            return '{"answer":"unknown","brief_explanation":"Changing support changes the answer."}'
        raise AssertionError("Repair should not run when probes do not provide enough strong failures")

    pred, raw = symbcot_cage_predict(_example(), fake_call, _extract_label, _extract_premises, SymbCoTConfig(max_tokens=128, prompt_root=_prompt_root(tmp_path), max_counterfactuals=1))

    assert pred.method == "llm_symbcot_cage"
    assert pred.label == "true"
    assert [row["method"] for row in raw] == ["llm_symbcot_translation", "llm_symbcot_plan", "llm_symbcot"]
    assert any("counterfactual attribution probe" in call for call in calls)


def test_symbcot_cage_repairs_invalid_initial_answer(tmp_path):
    def fake_call(prompt, max_tokens, json_output, schema):
        if prompt.startswith("Translate"):
            return "Predicates:\nKind(alice)\nConclusion: Blue(alice)"
        if prompt.startswith("Plan"):
            return "Use F1 and R1."
        if prompt.startswith("Solve"):
            return "No final answer found."
        if "counterfactual attribution probe" in prompt:
            return '{"answer":"unknown","brief_explanation":"Probe result."}'
        if "CAGE repair layer" in prompt:
            return '{"answer":"true","causal_premises":["F1","R1"],"brief_explanation":"Repaired invalid SymbCoT output."}'
        raise AssertionError(prompt[:80])

    pred, raw = symbcot_cage_predict(_example(), fake_call, _extract_label, _extract_premises, SymbCoTConfig(max_tokens=128, prompt_root=_prompt_root(tmp_path), max_counterfactuals=1))

    assert pred.method == "llm_symbcot_cage"
    assert pred.label == "true"
    assert raw[-1]["method"] == "llm_symbcot_cage"


def test_folio_symbcot_cage_repairs_on_transfer_disagreement(tmp_path):
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append(prompt)
        if prompt.startswith("Translate"):
            return "Premises:\n∀x Kind(x) -> Blue(x)\nConclusion: Blue(alice)"
        if prompt.startswith("Plan"):
            return "Instantiate universal rule."
        if prompt.startswith("Solve"):
            return "Final answer: {false}"
        if "transfer probe" in prompt:
            return '{"answer":"true","brief_explanation":"The view disagrees with the SymbCoT answer."}'
        if "transfer repair layer" in prompt:
            return '{"answer":"true","causal_premises":["P1","P2"],"brief_explanation":"Repair follows the consistent views."}'
        raise AssertionError(prompt[:80])

    ex = FolioExample(
        "folio_x",
        ["All kind people are blue.", "Alice is kind."],
        ["∀x (Kind(x) -> Blue(x))", "Kind(alice)"],
        "Alice is blue.",
        "Blue(alice)",
        "true",
    )

    pred, raw = folio_symbcot_cage_predict(ex, fake_call, _extract_folio_label, SymbCoTConfig(max_tokens=128, prompt_root=_prompt_root(tmp_path)))

    assert pred.method == "llm_symbcot_cage"
    assert pred.label == "true"
    assert raw[-1]["method"] == "llm_symbcot_cage"
    assert any("weak consistency checks" in call for call in calls)


def test_folio_symbcot_predict_uses_folio_templates(tmp_path):
    def fake_call(prompt, max_tokens, json_output, schema):
        if prompt.startswith("Translate"):
            assert "All kind people are blue." in prompt
            assert "Alice is blue." in prompt
            return "Premises:\n∀x Kind(x) -> Blue(x)\nConclusion: Blue(alice)"
        if prompt.startswith("Plan"):
            return "Instantiate universal rule."
        return "Final answer: {entailment}"

    ex = FolioExample(
        "folio_x",
        ["All kind people are blue.", "Alice is kind."],
        ["∀x (Kind(x) -> Blue(x))", "Kind(alice)"],
        "Alice is blue.",
        "Blue(alice)",
        "true",
    )

    pred, raw = folio_symbcot_predict(ex, fake_call, _extract_folio_label, SymbCoTConfig(max_tokens=128, prompt_root=_prompt_root(tmp_path)))

    assert pred.method == "llm_symbcot"
    assert pred.label == "true"
    assert [row["method"] for row in raw] == ["llm_symbcot_translation", "llm_symbcot_plan", "llm_symbcot"]
