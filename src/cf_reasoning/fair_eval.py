from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_LLM_METHODS = (
    "llm_direct",
    "llm_cpa",
    "llm_logiclm",
    "llm_symbcot",
    "llm_vericot",
    "llm_cage",
    "llm_cage_select",
    "llm_symbcot_cage",
)

_METHOD_ALIASES = {
    "folio_llm_direct": "llm_direct",
    "folio_llm_cpa": "llm_cpa",
    "folio_llm_logiclm": "llm_logiclm",
    "folio_llm_symbcot": "llm_symbcot",
    "folio_llm_vericot": "llm_vericot",
    "folio_llm_cage": "llm_cage",
    "folio_llm_cage_select": "llm_cage_select",
    "folio_llm_direct_cage": "llm_direct_cage",
    "folio_llm_logiclm_cage": "llm_logiclm_cage",
    "folio_llm_symbcot_cage": "llm_symbcot_cage",
}


def normalize_method_name(method: str) -> str:
    return _METHOD_ALIASES.get(method, method)


def load_canonical_llm_results(path: str | Path, benchmark: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method" not in df.columns:
        raise ValueError(f"{path} is missing required column: method")
    if "accuracy" not in df.columns:
        raise ValueError(f"{path} is missing required column: accuracy")
    out = df.copy()
    out["benchmark"] = out.get("benchmark", benchmark)
    out["method"] = out["method"].map(normalize_method_name)
    if "split" not in out.columns:
        out["split"] = benchmark
    cols = ["benchmark", "split", "method", "example_id", "gold", "pred", "accuracy"]
    return out[[col for col in cols if col in out.columns] + [col for col in out.columns if col not in cols]]


def summarize_fair_method_coverage(df: pd.DataFrame, required_methods: tuple[str, ...] = REQUIRED_LLM_METHODS) -> pd.DataFrame:
    rows = []
    for benchmark, group in df.groupby("benchmark", dropna=False):
        present = set(group["method"])
        rows.append(
            {
                "benchmark": benchmark,
                "num_examples": int(group["example_id"].nunique()) if "example_id" in group.columns else 0,
                "present_methods": ",".join(method for method in required_methods if method in present),
                "missing_methods": ",".join(method for method in required_methods if method not in present),
                "is_fair_method_set": all(method in present for method in required_methods),
            }
        )
    return pd.DataFrame(rows)


def cross_benchmark_accuracy(df: pd.DataFrame, required_methods: tuple[str, ...] = REQUIRED_LLM_METHODS) -> pd.DataFrame:
    fair = df[df["method"].isin(required_methods)].copy()
    if fair.empty:
        return pd.DataFrame(columns=["method"])
    table = fair.groupby(["method", "benchmark"], dropna=False)["accuracy"].mean().unstack("benchmark").reset_index()
    ordered = {method: i for i, method in enumerate(required_methods)}
    table["_order"] = table["method"].map(ordered)
    table = table.sort_values("_order").drop(columns="_order")
    return table


def write_fair_eval_tables(result_paths: dict[str, str | Path], output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [load_canonical_llm_results(path, benchmark) for benchmark, path in result_paths.items()]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    coverage = summarize_fair_method_coverage(combined)
    accuracy = cross_benchmark_accuracy(combined)
    combined.to_csv(output / "results_llm_all_benchmarks.csv", index=False)
    coverage.to_csv(output / "fair_method_coverage.csv", index=False)
    accuracy.to_csv(output / "cross_benchmark_accuracy.csv", index=False)
    return coverage, accuracy
