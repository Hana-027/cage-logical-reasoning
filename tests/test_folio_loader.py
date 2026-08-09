from cf_reasoning.datasets import load_folio_examples


def test_load_folio_validation_examples():
    examples = load_folio_examples("data/raw/FOLIO/data/v0.0/folio-validation.jsonl", limit=5, split="folio")
    assert len(examples) == 5
    assert {ex.label for ex in examples}.issubset({"true", "false", "unknown"})
    assert all(ex.premises for ex in examples)
    assert all(ex.conclusion for ex in examples)
    assert all("Conclusion:" in ex.text for ex in examples)
