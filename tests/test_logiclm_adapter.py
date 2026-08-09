from pathlib import Path

from cf_reasoning.datasets.folio import FolioExample
from cf_reasoning.llm_client import _extract_folio_label, _extract_label, _extract_premises
from cf_reasoning.logiclm_adapter import LogicLMConfig, folio_logiclm_predict, logiclm_predict
from cf_reasoning.schema import Example, Fact, Literal, Rule


def _prompt_root(tmp_path: Path) -> Path:
    tmp_path.mkdir(exist_ok=True)
    for dataset in ["ProntoQA", "ProofWriter", "FOLIO"]:
        (tmp_path / f"{dataset}.txt").write_text("Logic prompt [[PROBLEM]] / [[QUESTION]]", encoding="utf-8")
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
        split="prontoqa",
    )


def test_logiclm_predict_executes_generated_structured_program(tmp_path):
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append((prompt, json_output))
        return """Predicates:
Kind($x, bool) ::: Is x kind?
Blue($x, bool) ::: Is x blue?

Facts:
Kind(alice, True)

Rules:
Kind($x, True) >>> Blue($x, True)

Query:
Blue(alice, True)"""

    pred, raw = logiclm_predict(_example(), fake_call, _extract_label, _extract_premises, LogicLMConfig(max_tokens=128, prompt_root=_prompt_root(tmp_path)))

    assert pred.method == "llm_logiclm"
    assert pred.label == "true"
    assert pred.premise_ids == ("F1", "R1")
    assert [row["method"] for row in raw] == ["llm_logiclm_program"]
    assert calls == [("Logic prompt F1: Alice is kind.\nR1: If someone is kind, then they are blue. / Is it true that Alice is blue?", False)]


def test_logiclm_predict_falls_back_when_program_unparseable(tmp_path):
    calls = []

    def fake_call(prompt, max_tokens, json_output, schema):
        calls.append((prompt, json_output))
        if not json_output:
            return "not a valid program"
        return '{"answer":"false","causal_premises":["F1"],"brief_explanation":"Fallback answer."}'

    pred, raw = logiclm_predict(_example(), fake_call, _extract_label, _extract_premises, LogicLMConfig(max_tokens=128, prompt_root=_prompt_root(tmp_path)))

    assert pred.method == "llm_logiclm"
    assert pred.label == "false"
    assert pred.premise_ids == ("F1",)
    assert [row["method"] for row in raw] == ["llm_logiclm_program", "llm_logiclm"]
    assert calls[-1][1] is True


def test_logiclm_parser_handles_comments_and_multi_antecedent_rules(tmp_path):
    program = """Predicates:
Cold($x, bool) ::: Is x cold?
Quiet($x, bool) ::: Is x quiet?
Smart($x, bool) ::: Is x smart?

Facts:
Cold(Bob, True) ::: Bob is cold.
Quiet(Bob, True) ::: Bob is quiet.

Rules:
Quiet($x, True) && Cold($x, True) >>> Smart($x, True) ::: Quiet and cold things are smart.

Query:
Smart(Bob, True) ::: Bob is smart."""

    pred, raw = logiclm_predict(
        _example(),
        lambda prompt, max_tokens, json_output, schema: program,
        _extract_label,
        _extract_premises,
        LogicLMConfig(max_tokens=128, prompt_root=_prompt_root(tmp_path)),
    )

    assert pred.method == "llm_logiclm"
    assert pred.label == "true"
    assert [row["method"] for row in raw] == ["llm_logiclm_program"]


def test_folio_logiclm_predict_uses_folio_prompt_and_fallback(tmp_path):
    def fake_call(prompt, max_tokens, json_output, schema):
        if not json_output:
            assert "All kind people are blue." in prompt
            assert "Alice is blue." in prompt
            return "FOL program"
        return '{"answer":"entailment","brief_explanation":"The formalized rule entails the conclusion."}'

    ex = FolioExample(
        "folio_x",
        ["All kind people are blue.", "Alice is kind."],
        ["∀x (Kind(x) -> Blue(x))", "Kind(alice)"],
        "Alice is blue.",
        "Blue(alice)",
        "true",
    )

    pred, raw = folio_logiclm_predict(ex, fake_call, _extract_folio_label, LogicLMConfig(max_tokens=128, prompt_root=_prompt_root(tmp_path)))

    assert pred.method == "llm_logiclm"
    assert pred.label == "true"
    assert [row["method"] for row in raw] == ["llm_logiclm_program", "llm_logiclm"]
