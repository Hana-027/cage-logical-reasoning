from cf_reasoning.llm_client import _extract_folio_label, _extract_label, _extract_premises


def test_extract_label_accepts_bare_json_bool():
    assert _extract_label("true") == "true"
    assert _extract_label("false") == "false"


def test_extract_label_accepts_json_object_inside_text():
    raw = 'Here is the result: {"answer": "unknown", "causal_premises": ["F1"]}'
    assert _extract_label(raw) == "unknown"
    assert _extract_premises(raw) == ("F1",)


def test_general_label_parser_accepts_binary_aliases():
    assert _extract_label('{"answer": "yes", "causal_premises": ["F1"]}') == "true"
    assert _extract_label('{"answer": "No", "causal_premises": ["F1"]}') == "false"
    assert _extract_label('{"answer": false, "proof": "..."}') == "false"


def test_general_label_parser_accepts_alternate_answer_keys():
    assert _extract_label('{"label": "unknown", "proof": "..."}') == "unknown"
    assert _extract_label('{"final_label": "False"}') == "false"
    assert _extract_label('{"final_label": "entailed"}') == "true"
    assert _extract_label('{"verdict": "not entailed"}') == "unknown"


def test_general_label_parser_handles_nested_revised_answer():
    raw = '{"label": "invalid", "revised_answer": {"answer": "No", "proof": "..."}}'
    assert _extract_label(raw) == "false"


def test_general_label_parser_ignores_intermediate_containers():
    assert _extract_label('{"primary_strategy": "backward_chaining", "check_negation": true}') == "invalid"
    assert _extract_label('{"candidates": [{"answer": "true"}]}') == "invalid"


def test_folio_label_parser_accepts_strategy_aliases():
    assert _extract_folio_label('{"label": "entailment"}') == "true"
    assert _extract_folio_label('{"label": "entailed"}') == "true"
    assert _extract_folio_label('{"final_label": "entails"}') == "true"
    assert _extract_folio_label('{"prediction": "contradictory"}') == "false"


def test_folio_label_parser_accepts_unknown_aliases():
    assert _extract_folio_label('{"label": "unsupported"}') == "unknown"
    assert _extract_folio_label('{"label": "not_entailed"}') == "unknown"
    assert _extract_folio_label('{"label": "not entailed"}') == "unknown"
    assert _extract_folio_label('{"label": "neutral"}') == "unknown"
    assert _extract_folio_label('{"label": "inconclusive"}') == "unknown"
    assert _extract_folio_label('{"label": "uncertain"}') == "unknown"


def test_folio_label_parser_handles_validity_keys_cautiously():
    assert _extract_folio_label('{"validity": "valid"}') == "true"
    assert _extract_folio_label('{"validity": "invalid"}') == "unknown"
    assert _extract_folio_label('{"valid": true}') == "true"
    assert _extract_folio_label('{"valid": false}') == "unknown"
    assert _extract_folio_label('{"conclusion_entailed": true}') == "true"
    assert _extract_folio_label('{"conclusion_entailed": false}') == "unknown"


def test_folio_label_parser_refuses_ambiguous_numeric_labels():
    assert _extract_folio_label('{"label": "0"}') == "invalid"


def test_folio_label_parser_reads_final_long_text_answer():
    raw = "The premise discussion mentions unknown cases, but the conclusion follows. Final answer: **true**"
    assert _extract_folio_label(raw) == "true"


def test_general_label_parser_reads_conclusion_sentence_in_json_proof():
    raw = '{"proof": "The derivation yields opaque, not not opaque. Therefore, the query is false."}'
    assert _extract_label(raw) == "false"
