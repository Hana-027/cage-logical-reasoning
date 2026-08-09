from cf_reasoning.fair_eval import REQUIRED_LLM_METHODS
from cf_reasoning.preflight import build_preflight_report, write_preflight_report


def test_build_preflight_report_loads_three_benchmarks():
    report = build_preflight_report(n_examples=5)

    assert report["fairness_ready"] is True
    assert set(report["datasets"]) == {"proofwriter", "prontoqa", "folio"}
    assert report["datasets"]["proofwriter"]["parsed"] == 5
    assert report["datasets"]["prontoqa"]["parsed"] == 5
    assert report["datasets"]["folio"]["count"] == 5
    assert "llm_symbcot_cage" in report["method_set"]["required_result_methods"]
    assert tuple(report["method_set"]["required_result_methods"]) == REQUIRED_LLM_METHODS


def test_write_preflight_report(tmp_path):
    report = build_preflight_report(n_examples=3)
    write_preflight_report(report, tmp_path)

    assert (tmp_path / "preflight_report.json").exists()
    assert (tmp_path / "preflight_datasets.csv").exists()
