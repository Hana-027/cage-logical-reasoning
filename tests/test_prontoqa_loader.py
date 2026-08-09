from cf_reasoning.datasets import load_prontoqa_examples


def test_load_prontoqa_examples_from_zip():
    examples, report = load_prontoqa_examples("data/raw/prontoqa/generated_ood_data.zip", limit=10, split="test")
    assert len(examples) == 10
    assert report.parsed == 10
    assert all(ex.label == "true" for ex in examples)
    assert all(ex.facts for ex in examples)
    assert all(ex.rules for ex in examples)
    assert all(ex.support_ids for ex in examples)


def test_load_prontoqa2_json_preserves_true_and_false_labels():
    examples, report = load_prontoqa_examples("data/raw/prontoqa2/ProntoQA_dev_gpt-4.json", limit=20, split="dev")

    assert len(examples) == 20
    assert report.parsed == 20
    assert report.skipped == 0
    assert {ex.label for ex in examples} == {"true", "false"}
    assert report.label_counts["true"] > 0
    assert report.label_counts["false"] > 0
    assert all(ex.split == "dev" for ex in examples)
    assert all(ex.facts for ex in examples)
    assert all(ex.rules for ex in examples)


def test_load_prontoqa2_parses_question_polarity():
    examples, _ = load_prontoqa_examples("data/raw/prontoqa2/ProntoQA_dev_gpt-4.json", limit=5, split="dev")
    by_id = {ex.id: ex for ex in examples}

    ex1 = by_id["pronto_ProntoQA_1"]
    assert ex1.query.entity == "max"
    assert ex1.query.predicate == "sour"
    assert ex1.query.negated is False
    assert ex1.label == "false"

    ex3 = by_id["pronto_ProntoQA_3"]
    assert ex3.query.entity == "wren"
    assert ex3.query.predicate == "metallic"
    assert ex3.query.negated is True
    assert ex3.label == "true"
