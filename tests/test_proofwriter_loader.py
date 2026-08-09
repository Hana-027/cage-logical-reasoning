from pathlib import Path

from cf_reasoning.datasets import load_proofwriter_examples


def test_load_proofwriter_style_jsonl(tmp_path: Path):
    path = tmp_path / "proofwriter_sample.jsonl"
    path.write_text(
        '{"id":"1","theory":"F1: Alice is kind. R1: If Alice is kind then Alice is blue.","question":"Alice is blue.","answer":"true","depth":1}\n'
        '{"id":"2","theory":"Bob is red.","question":"Bob is blue.","answer":"unknown","depth":0}\n',
        encoding="utf-8",
    )

    examples, report = load_proofwriter_examples(path)

    assert report.loaded == 2
    assert report.parsed == 2
    assert examples[0].label == "true"
    assert examples[0].support_ids == ("F1", "R1")
    assert examples[1].label == "unknown"


def test_load_translation_format_jsonl(tmp_path: Path):
    path = tmp_path / "proofwriter_translation.jsonl"
    path.write_text(
        '{"translation":{"en":"$answer$ ; $proof$ ; $question$ = Fiona is cold. ; $context$ = sent1: Fiona is young. sent2: Fiona is white. sent3: All young, white things are cold.","ro":"$answer$ = True ; $proof$ = sent3 sent1 sent2"}}\n',
        encoding="utf-8",
    )

    examples, report = load_proofwriter_examples(path, split="dev")

    assert report.loaded == 1
    assert report.parsed == 1
    assert report.to_dict("dev")["split"] == "dev"
    assert examples[0].label == "true"
    assert examples[0].split == "dev"
    assert set(examples[0].support_ids) == {"SENT1", "SENT2", "SENT3"}


def test_load_people_pronoun_and_relation_formats(tmp_path: Path):
    path = tmp_path / "proofwriter_richer.jsonl"
    path.write_text(
        '{"translation":{"en":"$answer$ ; $question$ = The cat eats the mouse. ; $context$ = sent1: The cat chases the mouse. sent2: If someone chases the mouse then they eat the mouse.","ro":"$answer$ = True"}}\n'
        '{"translation":{"en":"$answer$ ; $question$ = Bob is rough. ; $context$ = sent1: Bob is big. sent2: All big people are rough.","ro":"$answer$ = True"}}\n'
        '{"translation":{"en":"$answer$ ; $question$ = Gary is green. ; $context$ = sent1: Gary is rough. sent2: If someone is rough then they are green.","ro":"$answer$ = True"}}\n'
        '{"translation":{"en":"$answer$ ; $question$ = The cow does not need the rabbit. ; $context$ = sent1: The cow needs the rabbit.","ro":"$answer$ = False"}}\n'
        '{"translation":{"en":"$answer$ ; $question$ = The bald eagle chases the bald eagle. ; $context$ = sent1: The bald eagle chases the lion. sent2: If someone chases the lion then they chase the bald eagle.","ro":"$answer$ = True"}}\n',
        encoding="utf-8",
    )

    examples, report = load_proofwriter_examples(path)

    assert report.parsed == 5
    assert examples[0].query.predicate == "eat__mouse"
    assert set(examples[0].support_ids) == {"SENT1", "SENT2"}
    assert set(examples[1].support_ids) == {"SENT1", "SENT2"}
    assert examples[2].query.predicate == "green"
    assert examples[3].query.negated is True
