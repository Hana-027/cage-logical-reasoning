import pandas as pd

from cf_reasoning.fair_eval import REQUIRED_LLM_METHODS, cross_benchmark_accuracy, load_canonical_llm_results, summarize_fair_method_coverage


def test_load_canonical_llm_results_normalizes_legacy_folio_alias(tmp_path):
    path = tmp_path / "results_folio_llm.csv"
    pd.DataFrame(
        [
            {"split": "folio", "example_id": "folio_1", "method": "folio_llm_logiclm", "gold": "true", "pred": "true", "accuracy": 1},
            {"split": "folio", "example_id": "folio_1", "method": "llm_cage", "gold": "true", "pred": "unknown", "accuracy": 0},
        ]
    ).to_csv(path, index=False)

    out = load_canonical_llm_results(path, "folio")

    assert list(out.columns[:7]) == ["benchmark", "split", "method", "example_id", "gold", "pred", "accuracy"]
    assert set(out["method"]) == {"llm_logiclm", "llm_cage"}
    assert set(out["benchmark"]) == {"folio"}


def test_fair_method_coverage_reports_missing_methods():
    df = pd.DataFrame(
        [
            {"benchmark": "proofwriter", "split": "test", "example_id": "ex1", "method": method, "accuracy": 1}
            for method in REQUIRED_LLM_METHODS[:-1]
        ]
    )

    coverage = summarize_fair_method_coverage(df)

    assert coverage.iloc[0]["is_fair_method_set"] == False
    assert coverage.iloc[0]["missing_methods"] == "llm_symbcot_cage"


def test_cross_benchmark_accuracy_uses_required_methods_only():
    rows = []
    for benchmark in ["proofwriter", "prontoqa", "folio"]:
        for method in REQUIRED_LLM_METHODS:
            rows.append({"benchmark": benchmark, "split": benchmark, "example_id": f"{benchmark}_1", "method": method, "accuracy": 1.0})
        rows.append({"benchmark": benchmark, "split": benchmark, "example_id": f"{benchmark}_1", "method": "llm_guided_mpss", "accuracy": 0.0})
    df = pd.DataFrame(rows)

    table = cross_benchmark_accuracy(df)

    assert table["method"].tolist() == list(REQUIRED_LLM_METHODS)
    assert "llm_guided_mpss" not in set(table["method"])
    assert set(table.columns) == {"method", "proofwriter", "prontoqa", "folio"}
