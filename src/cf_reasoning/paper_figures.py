from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .fair_eval import normalize_method_name

VALID_LABELS = {"true", "false", "unknown"}
TRACE_METHODS = {
    "llm_direct_cage",
    "llm_logiclm_cage",
    "llm_symbcot_cage",
    "llm_direct_cage_gated",
    "llm_logiclm_cage_gated",
    "llm_symbcot_cage_gated",
}
CAGE_METHODS = (
    "llm_direct_cage",
    "llm_direct_cage_gated",
    "llm_logiclm_cage",
    "llm_logiclm_cage_gated",
    "llm_symbcot_cage",
    "llm_symbcot_cage_gated",
)
METHOD_ORDER = (
    "llm_direct",
    "llm_direct_cage",
    "llm_direct_cage_gated",
    "llm_logiclm",
    "llm_logiclm_cage",
    "llm_logiclm_cage_gated",
    "llm_symbcot",
    "llm_symbcot_cage",
    "llm_symbcot_cage_gated",
)
METHOD_LABELS = {
    "llm_direct": "Direct",
    "llm_direct_cage": "Direct+CAGE",
    "llm_direct_cage_gated": "Direct+CAGE-Gated",
    "llm_logiclm": "Logic-LM",
    "llm_logiclm_cage": "Logic-LM+CAGE",
    "llm_logiclm_cage_gated": "Logic-LM+CAGE-Gated",
    "llm_symbcot": "SymbCoT",
    "llm_symbcot_cage": "SymbCoT+CAGE",
    "llm_symbcot_cage_gated": "SymbCoT+CAGE-Gated",
}
METHOD_COLORS = {
    "llm_direct": "#4C78A8",
    "llm_direct_cage": "#9ECAE1",
    "llm_direct_cage_gated": "#2F5597",
    "llm_logiclm": "#59A14F",
    "llm_logiclm_cage": "#A7D9A0",
    "llm_logiclm_cage_gated": "#2E7D32",
    "llm_symbcot": "#F28E2B",
    "llm_symbcot_cage": "#F6C98D",
    "llm_symbcot_cage_gated": "#C55A11",
}


@dataclass(frozen=True)
class RunBundle:
    benchmark: str
    path: Path
    base: pd.DataFrame
    counterfactual: pd.DataFrame
    predictions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready tables and figures from LLM experiment output directories.")
    parser.add_argument("runs", nargs="*", help="Output directories, optionally benchmark=path.")
    parser.add_argument("--run", action="append", default=[], help="Output directory, optionally benchmark=path. May be repeated.")
    parser.add_argument("--output-dir", default="outputs/paper_figures", help="Directory for generated tables and figures.")
    parser.add_argument("--no-figures", action="store_true", help="Only write CSV/LaTeX tables.")
    return parser.parse_args()


def build_paper_outputs(run_specs: Iterable[str], output_dir: str | Path, make_figures: bool = True) -> dict[str, pd.DataFrame]:
    specs = list(run_specs)
    if not specs:
        raise ValueError("Pass at least one experiment output directory with --run, e.g. --run prontoqa=outputs/prontoqa2_100_main_v2")
    bundles = [load_run(spec) for spec in specs]
    methods = METHOD_ORDER
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if make_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    base = _concat([bundle.base for bundle in bundles])
    counterfactual = _concat([bundle.counterfactual for bundle in bundles])
    predictions = _concat([bundle.predictions for bundle in bundles])
    benchmark_order = [bundle.benchmark for bundle in bundles]

    coverage = method_coverage(base, methods)
    accuracy = accuracy_table(base, methods)
    invalid = invalid_rate_table(base, methods)
    symbcot = symbcot_cage_comparison(base)
    repair = repair_summary(base, predictions)
    diagnostics = diagnostic_summary(predictions)
    diagnostic_breakdown = diagnostic_failure_breakdown(predictions)
    cf_metrics = counterfactual_metric_table(counterfactual, methods)
    cf_family = counterfactual_family_table(counterfactual, methods)

    outputs = {
        "combined_results_llm": base,
        "combined_counterfactual_results_llm": counterfactual,
        "method_coverage": coverage,
        "accuracy_table": accuracy,
        "invalid_rate_table": invalid,
        "symbcot_cage_comparison": symbcot,
        "repair_summary": repair,
        "diagnostic_summary": diagnostics,
        "diagnostic_failure_breakdown": diagnostic_breakdown,
        "counterfactual_metric_table": cf_metrics,
        "counterfactual_family_table": cf_family,
    }
    for name, table in outputs.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    _write_percent_latex(accuracy, tables_dir / "accuracy_table.tex", value_cols=[c for c in accuracy.columns if c != "method"])
    _write_percent_latex(invalid, tables_dir / "invalid_rate_table.tex", value_cols=[c for c in invalid.columns if c != "method"])
    _write_percent_latex(symbcot, tables_dir / "symbcot_cage_comparison.tex", value_cols=[c for c in symbcot.columns if c not in {"benchmark", "n_examples"}])
    _write_percent_latex(repair, tables_dir / "repair_summary.tex", value_cols=[c for c in repair.columns if c not in {"benchmark", "method", "n_examples"}])
    _write_percent_latex(diagnostics, tables_dir / "diagnostic_summary.tex", value_cols=[c for c in diagnostics.columns if c not in {"benchmark", "method", "n_examples", "n_probes"}])

    if make_figures:
        plot_metric_by_benchmark(base, "accuracy", methods, benchmark_order, figures_dir / "accuracy_by_benchmark", "Answer accuracy by benchmark", "Accuracy")
        plot_metric_by_benchmark(_with_invalid_indicator(base), "invalid_rate", methods, benchmark_order, figures_dir / "invalid_rate_by_benchmark", "Invalid answer rate by benchmark", "Invalid rate")
        plot_symbcot_comparison(symbcot, figures_dir / "symbcot_vs_symbcot_cage")
        plot_repair_gain_harm(repair, figures_dir / "repair_gain_harm")
        plot_diagnostic_pass_rate(diagnostics, figures_dir / "diagnostic_pass_rate")
        plot_algorithm_diagram(figures_dir / "cage_principle")

    return outputs


def load_run(spec: str) -> RunBundle:
    benchmark_hint, path = _parse_run_spec(spec)
    result_path = _first_existing(path, ["results_llm.csv", "results_folio_llm.csv", "results_base_llm.csv"])
    if result_path is None:
        raise FileNotFoundError(f"No LLM result CSV found in {path}")
    base = pd.read_csv(result_path)
    benchmark = benchmark_hint or _infer_benchmark(base, path)
    base = _canonicalize_results(base, benchmark)

    cf_path = path / "results_counterfactual_llm.csv"
    counterfactual = pd.read_csv(cf_path) if cf_path.exists() else pd.DataFrame()
    if not counterfactual.empty:
        counterfactual = _canonicalize_results(counterfactual, benchmark, require_accuracy=False)

    pred_path = path / "llm_predictions.jsonl"
    predictions = _read_predictions(pred_path, benchmark) if pred_path.exists() else pd.DataFrame()
    return RunBundle(benchmark, path, base, counterfactual, predictions)


def method_coverage(base: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    if base.empty:
        return pd.DataFrame(columns=["benchmark", "n_examples", "present_methods", "missing_methods", "is_fair_method_set"])
    for benchmark, group in base.groupby("benchmark", dropna=False):
        present = set(group["method"])
        rows.append(
            {
                "benchmark": benchmark,
                "n_examples": int(group["example_id"].nunique()) if "example_id" in group.columns else 0,
                "present_methods": ",".join(method for method in methods if method in present),
                "missing_methods": ",".join(method for method in methods if method not in present),
                "is_fair_method_set": all(method in present for method in methods),
            }
        )
    return pd.DataFrame(rows).sort_values("benchmark").reset_index(drop=True)


def accuracy_table(base: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame(columns=["method"])
    table = _metric_pivot(base[base["method"].isin(methods)], "accuracy", methods)
    benchmark_cols = [c for c in table.columns if c != "method"]
    if benchmark_cols:
        table["mean"] = table[benchmark_cols].mean(axis=1, skipna=True)
    return table


def invalid_rate_table(base: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame(columns=["method"])
    return _metric_pivot(_with_invalid_indicator(base)[lambda d: d["method"].isin(methods)], "invalid_rate", methods)


def symbcot_cage_comparison(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if base.empty:
        return pd.DataFrame(columns=["benchmark", "n_examples", "symbcot_accuracy", "symbcot_cage_accuracy", "accuracy_delta", "symbcot_invalid_rate", "symbcot_cage_invalid_rate", "same_label_rate", "label_change_rate"])
    prepared = _with_invalid_indicator(base)
    for benchmark, group in prepared.groupby("benchmark", dropna=False):
        symb = group[group["method"] == "llm_symbcot"]
        cage = group[group["method"] == "llm_symbcot_cage"]
        paired = symb[["example_id", "pred_norm"]].merge(cage[["example_id", "pred_norm"]], on="example_id", suffixes=("_symbcot", "_cage"))
        rows.append(
            {
                "benchmark": benchmark,
                "n_examples": int(group["example_id"].nunique()),
                "symbcot_accuracy": _mean_or_nan(symb["accuracy"]),
                "symbcot_cage_accuracy": _mean_or_nan(cage["accuracy"]),
                "accuracy_delta": _mean_or_nan(cage["accuracy"]) - _mean_or_nan(symb["accuracy"]),
                "symbcot_invalid_rate": _mean_or_nan(symb["invalid_rate"]),
                "symbcot_cage_invalid_rate": _mean_or_nan(cage["invalid_rate"]),
                "same_label_rate": float((paired["pred_norm_symbcot"] == paired["pred_norm_cage"]).mean()) if not paired.empty else float("nan"),
                "label_change_rate": float((paired["pred_norm_symbcot"] != paired["pred_norm_cage"]).mean()) if not paired.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def repair_summary(base: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if base.empty or predictions.empty:
        return pd.DataFrame(columns=["benchmark", "method", "n_examples", "initial_accuracy", "final_accuracy", "accuracy_delta", "repair_triggered_rate", "answer_changed_rate", "repair_gain_rate", "repair_harm_rate", "net_repair_gain", "initial_invalid_rate", "final_invalid_rate"])
    base_cols = base[["benchmark", "method", "example_id", "gold", "pred", "accuracy"]].copy()
    trace_rows = []
    for _, row in predictions[predictions["method"].isin(TRACE_METHODS)].iterrows():
        trace = _decode_trace(row.get("raw_response"))
        if not trace:
            continue
        trace_rows.append(
            {
                "benchmark": row["benchmark"],
                "method": row["method"],
                "example_id": row["example_id"],
                "initial_answer": _normalize_label(trace.get("initial_answer") or trace.get("selected_initial_answer")),
                "final_answer": _normalize_label(trace.get("final_answer") or row.get("label")),
                "repair_triggered": _has_repair(trace, row["method"]),
            }
        )
    if not trace_rows:
        return pd.DataFrame()
    traces = pd.DataFrame(trace_rows)
    merged = traces.merge(base_cols, on=["benchmark", "method", "example_id"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["gold_norm"] = merged["gold"].map(_normalize_label)
    merged["pred_norm"] = merged["pred"].map(_normalize_label)
    merged["initial_accuracy"] = (merged["initial_answer"] == merged["gold_norm"]).astype(float)
    merged["final_accuracy"] = merged["accuracy"].astype(float)
    merged["answer_changed"] = (merged["initial_answer"] != merged["pred_norm"]).astype(float)
    merged["repair_gain"] = ((merged["initial_accuracy"] == 0) & (merged["final_accuracy"] == 1)).astype(float)
    merged["repair_harm"] = ((merged["initial_accuracy"] == 1) & (merged["final_accuracy"] == 0)).astype(float)
    merged["initial_invalid"] = (~merged["initial_answer"].isin(VALID_LABELS)).astype(float)
    merged["final_invalid"] = (~merged["pred_norm"].isin(VALID_LABELS)).astype(float)
    summary = (
        merged.groupby(["benchmark", "method"], dropna=False)
        .agg(
            n_examples=("example_id", "nunique"),
            initial_accuracy=("initial_accuracy", "mean"),
            final_accuracy=("final_accuracy", "mean"),
            repair_triggered_rate=("repair_triggered", "mean"),
            answer_changed_rate=("answer_changed", "mean"),
            repair_gain_rate=("repair_gain", "mean"),
            repair_harm_rate=("repair_harm", "mean"),
            initial_invalid_rate=("initial_invalid", "mean"),
            final_invalid_rate=("final_invalid", "mean"),
        )
        .reset_index()
    )
    summary["accuracy_delta"] = summary["final_accuracy"] - summary["initial_accuracy"]
    summary["net_repair_gain"] = summary["repair_gain_rate"] - summary["repair_harm_rate"]
    return _sort_methods(summary, CAGE_METHODS)


def diagnostic_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    diagnostics = _diagnostic_rows(predictions)
    if diagnostics.empty:
        return pd.DataFrame(columns=["benchmark", "method", "n_examples", "n_probes", "vcar", "failed_probe_rate", "invalid_probe_rate"])
    summary = (
        diagnostics.groupby(["benchmark", "method"], dropna=False)
        .agg(
            n_examples=("example_id", "nunique"),
            n_probes=("probe_id", "count"),
            vcar=("is_ok", "mean"),
            failed_probe_rate=("is_failed", "mean"),
            invalid_probe_rate=("is_invalid", "mean"),
        )
        .reset_index()
    )
    return _sort_methods(summary, CAGE_METHODS)


def diagnostic_failure_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    diagnostics = _diagnostic_rows(predictions)
    if diagnostics.empty:
        return pd.DataFrame(columns=["benchmark", "method", "diagnostic_status", "count", "rate"])
    counts = diagnostics.groupby(["benchmark", "method", "diagnostic_status"], dropna=False).size().reset_index(name="count")
    totals = counts.groupby(["benchmark", "method"], dropna=False)["count"].transform("sum")
    counts["rate"] = counts["count"] / totals
    return _sort_methods(counts, CAGE_METHODS)


def counterfactual_metric_table(counterfactual: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "counterfactual_consistency",
        "label_change_accuracy",
        "irrelevant_robustness",
        "attr_f1",
        "attribution_consistency",
        "support_shift_detected",
        "proof_break_sensitivity",
        "proof_preserve_robustness",
        "alternate_proof_awareness",
        "contradiction_detection",
    ]
    return _aggregate_metrics(counterfactual, ["benchmark", "method"], metrics, methods)


def counterfactual_family_table(counterfactual: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    metrics = ["accuracy", "counterfactual_consistency", "attr_f1", "attribution_consistency"]
    if counterfactual.empty or "cf_family" not in counterfactual.columns:
        return pd.DataFrame(columns=["benchmark", "method", "cf_family"] + metrics)
    return _aggregate_metrics(counterfactual, ["benchmark", "method", "cf_family"], metrics, methods)


def plot_metric_by_benchmark(df: pd.DataFrame, metric: str, methods: tuple[str, ...], benchmark_order: list[str], output_base: Path, title: str, ylabel: str) -> None:
    if df.empty or metric not in df.columns:
        return
    long = df[df["method"].isin(methods)].groupby(["benchmark", "method"], dropna=False)[metric].mean().reset_index()
    if long.empty:
        return
    benchmarks = [b for b in benchmark_order if b in set(long["benchmark"])] or sorted(long["benchmark"].unique())
    present_methods = [m for m in methods if m in set(long["method"])]
    fig, ax = plt.subplots(figsize=(max(6.8, 1.25 * len(benchmarks)), 4.2))
    _grouped_bars(ax, long, benchmarks, present_methods, metric)
    _finish_axis(ax, title, ylabel)
    ax.set_xticks(range(len(benchmarks)), benchmarks, rotation=20, ha="right")
    ax.legend([METHOD_LABELS.get(m, m) for m in present_methods], fontsize=8, ncol=2, frameon=False)
    _save_figure(fig, output_base)


def plot_symbcot_comparison(table: pd.DataFrame, output_base: Path) -> None:
    if table.empty:
        return
    plot_df = table.melt(id_vars="benchmark", value_vars=["symbcot_accuracy", "symbcot_cage_accuracy"], var_name="method", value_name="accuracy")
    plot_df["method"] = plot_df["method"].map({"symbcot_accuracy": "llm_symbcot", "symbcot_cage_accuracy": "llm_symbcot_cage"})
    benchmarks = list(table["benchmark"])
    fig, ax = plt.subplots(figsize=(max(6.4, 1.2 * len(benchmarks)), 4.0))
    _grouped_bars(ax, plot_df, benchmarks, ("llm_symbcot", "llm_symbcot_cage"), "accuracy")
    _finish_axis(ax, "SymbCoT vs. label-conservative SymbCoT+CAGE", "Accuracy")
    ax.set_xticks(range(len(benchmarks)), benchmarks, rotation=20, ha="right")
    ax.legend(["SymbCoT", "SymbCoT+CAGE"], fontsize=8, frameon=False)
    _save_figure(fig, output_base)


def plot_repair_gain_harm(repair: pd.DataFrame, output_base: Path) -> None:
    if repair.empty:
        return
    family_order = {
        "llm_direct_cage": 0,
        "llm_direct_cage_gated": 0,
        "llm_logiclm_cage": 1,
        "llm_logiclm_cage_gated": 1,
        "llm_symbcot_cage": 2,
        "llm_symbcot_cage_gated": 2,
    }
    variant_order = {
        "llm_direct_cage": 0,
        "llm_direct_cage_gated": 1,
        "llm_logiclm_cage": 0,
        "llm_logiclm_cage_gated": 1,
        "llm_symbcot_cage": 0,
        "llm_symbcot_cage_gated": 1,
    }
    plot = repair.copy()
    plot["_family_order"] = plot["method"].map(family_order).fillna(99)
    plot["_variant_order"] = plot["method"].map(variant_order).fillna(99)
    plot = plot.sort_values(["_family_order", "_variant_order", "benchmark", "method"]).reset_index(drop=True)
    labels = plot["method"].map(lambda m: METHOD_LABELS.get(m, m)) + "\n" + plot["benchmark"]
    x = range(len(plot))
    fig, ax = plt.subplots(figsize=(max(7.2, 0.82 * len(plot)), 4.3))
    ax.bar(x, plot["repair_gain_rate"], color="#0ca30c", label="Gain: wrong→correct")
    ax.bar(x, -plot["repair_harm_rate"], color="#d03b3b", label="Harm: correct→wrong")
    ax.axhline(0, color="#383835", linewidth=0.8)
    ax.set_xticks(list(x), labels, rotation=35, ha="right", fontsize=8)
    ax.set_title("Repair gain-harm balance by base algorithm")
    ax.set_ylabel("Share of examples")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    group_starts = plot.groupby("_family_order", sort=True).head(1).index.tolist()
    for start in group_starts[1:]:
        ax.axvline(start - 0.5, color="#c9c8c0", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    _save_figure(fig, output_base)


def plot_diagnostic_pass_rate(diagnostics: pd.DataFrame, output_base: Path) -> None:
    if diagnostics.empty:
        return
    benchmarks = sorted(diagnostics["benchmark"].unique())
    methods = [m for m in CAGE_METHODS if m in set(diagnostics["method"])]
    fig, ax = plt.subplots(figsize=(max(6.8, 1.1 * len(benchmarks)), 4.0))
    _grouped_bars(ax, diagnostics, benchmarks, methods, "vcar")
    _finish_axis(ax, "Verifier counterfactual acceptance rate", "VCAR / diagnostic pass rate")
    ax.set_xticks(range(len(benchmarks)), benchmarks, rotation=20, ha="right")
    ax.legend([METHOD_LABELS.get(m, m) for m in methods], fontsize=8, frameon=False)
    _save_figure(fig, output_base)


def plot_algorithm_diagram(output_base: Path) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def panel(x, y, w, h, label, title, face="#fbfaf7"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=1.0, edgecolor="#2f2f2c", facecolor=face)
        ax.add_patch(patch)
        ax.text(x + 0.012, y + h - 0.035, label, fontsize=11, fontweight="bold", ha="left", va="center")
        ax.text(x + 0.045, y + h - 0.035, title, fontsize=10, fontweight="bold", ha="left", va="center")

    def box(x, y, w, h, text, face, fontsize=8.4, edge="#b9b8af", weight="normal"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.012", linewidth=0.9, edgecolor=edge, facecolor=face)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color="#111111")

    def arrow(start, end, color="#52514e", lw=1.0, rad=0.0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}"))

    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel(0.02, 0.52, 0.23, 0.40, "a)", "Base reasoning")
    box(0.045, 0.79, 0.18, 0.065, "Context + query", "#f0efec", weight="bold")
    box(0.045, 0.69, 0.052, 0.06, "Direct", "#d8e8fb")
    box(0.109, 0.69, 0.052, 0.06, "Logic-LM", "#d8e8fb")
    box(0.173, 0.69, 0.052, 0.06, "SymbCoT", "#d8e8fb")
    box(0.052, 0.58, 0.166, 0.07, "initial label\n+ claimed support", "#fff3cf")
    arrow((0.135, 0.79), (0.135, 0.75))
    arrow((0.071, 0.69), (0.112, 0.65))
    arrow((0.135, 0.69), (0.135, 0.65))
    arrow((0.199, 0.69), (0.158, 0.65))

    panel(0.29, 0.52, 0.32, 0.40, "b)", "Counterfactual attribution diagnostics")
    probes = [
        ("Proof-breaking", "necessary premise removed", "label should change", "#f7d8e5"),
        ("Proof-preserving", "alternative proof remains", "label should stay", "#eadcf6"),
        ("Support-shift", "proof path changes", "support should shift", "#e1edf9"),
        ("Distractor", "irrelevant fact perturbed", "label should stay", "#e7f3df"),
        ("Contradiction", "conflicting premise added", "unknown / conflict", "#fde1e1"),
    ]
    for i, (name, edit, expected, color) in enumerate(probes):
        y = 0.82 - i * 0.055
        box(0.315, y, 0.092, 0.038, name, color, fontsize=7.4, weight="bold")
        ax.text(0.42, y + 0.019, edit, fontsize=7.3, va="center", ha="left")
        ax.text(0.535, y + 0.019, expected, fontsize=7.3, va="center", ha="left", color="#4a3aa7")
    box(0.365, 0.575, 0.17, 0.06, "diagnostic trace\npass / fail + failure type", "#f0efec", fontsize=8.0)
    arrow((0.25, 0.62), (0.29, 0.69), rad=-0.12)

    panel(0.65, 0.52, 0.33, 0.40, "c)", "Repair decision")
    box(0.685, 0.80, 0.115, 0.06, "Is initial label\nvalid?", "#fff3cf", fontsize=8.0, weight="bold")
    box(0.69, 0.66, 0.12, 0.07, "Conservative\npreserve valid label", "#dcf5ec", fontsize=7.9)
    box(0.84, 0.66, 0.11, 0.07, "Gated\nrepair if diagnostics agree", "#e2def8", fontsize=7.7)
    box(0.765, 0.555, 0.12, 0.06, "Invalid-output\nrepair", "#fde1e1", fontsize=7.9)
    arrow((0.742, 0.80), (0.742, 0.73))
    arrow((0.80, 0.76), (0.87, 0.73), rad=-0.12)
    arrow((0.742, 0.80), (0.815, 0.615), rad=-0.2)
    ax.text(0.714, 0.745, "valid", fontsize=7.5, color="#52514e")
    ax.text(0.86, 0.755, "strong\nagreement", fontsize=7.2, color="#52514e")
    ax.text(0.77, 0.665, "invalid", fontsize=7.5, color="#52514e")
    arrow((0.61, 0.62), (0.65, 0.70), rad=-0.1)

    panel(0.08, 0.08, 0.84, 0.32, "d)", "Evaluation outputs")
    box(0.12, 0.26, 0.105, 0.06, "Answer\naccuracy", "#f0efec", fontsize=8.0)
    box(0.25, 0.26, 0.105, 0.06, "Invalid\nrate", "#f0efec", fontsize=8.0)
    box(0.38, 0.26, 0.13, 0.06, "Counterfactual\nrobustness", "#f0efec", fontsize=8.0)
    box(0.535, 0.26, 0.12, 0.06, "Attribution\nquality", "#f0efec", fontsize=8.0)
    box(0.68, 0.26, 0.13, 0.06, "Repair\nreliability", "#f0efec", fontsize=8.0)
    box(0.31, 0.13, 0.38, 0.065, "Final label + support + diagnostic trace + repair audit", "#fff7cf", fontsize=8.5, weight="bold")
    for x in [0.172, 0.302, 0.445, 0.595, 0.745]:
        arrow((x, 0.26), (0.50, 0.195), rad=0.08 if x < 0.5 else -0.08)
    arrow((0.80, 0.555), (0.72, 0.40), rad=0.1)
    arrow((0.50, 0.575), (0.50, 0.40), rad=0.0)

    ax.text(0.02, 0.965, "CAGE: Counterfactual Attribution-Guided Evaluation and Repair", fontsize=14, fontweight="bold", ha="left")
    ax.text(0.02, 0.935, "A method-agnostic wrapper for testing whether logical answers depend on the premises that causally support them.", fontsize=9.5, ha="left", color="#444441")
    fig.tight_layout(pad=0.4)
    _save_figure(fig, output_base)


def _parse_run_spec(spec: str) -> tuple[str | None, Path]:
    if "=" in spec:
        benchmark, raw_path = spec.split("=", 1)
        return benchmark.strip(), Path(raw_path.strip())
    return None, Path(spec)


def _first_existing(path: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def _canonicalize_results(df: pd.DataFrame, benchmark: str, require_accuracy: bool = True) -> pd.DataFrame:
    out = df.copy()
    if "method" not in out.columns:
        raise ValueError("Result CSV is missing required column: method")
    if require_accuracy and "accuracy" not in out.columns:
        raise ValueError("Result CSV is missing required column: accuracy")
    out["benchmark"] = benchmark
    out["method"] = out["method"].map(lambda m: normalize_method_name(str(m)))
    if "split" not in out.columns:
        out["split"] = benchmark
    return out


def _infer_benchmark(df: pd.DataFrame, path: Path) -> str:
    if "benchmark" in df.columns:
        values = [str(v) for v in df["benchmark"].dropna().unique()]
        if len(values) == 1:
            return values[0]
    name = path.name.lower()
    if "proofwriter" in name:
        return "proofwriter"
    if "pronto" in name:
        return "prontoqa"
    if "folio" in name:
        return "folio"
    if "split" in df.columns and not df["split"].dropna().empty:
        return str(df["split"].dropna().iloc[0])
    return path.name


def _read_predictions(path: Path, benchmark: str) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["benchmark"] = benchmark
            row["method"] = normalize_method_name(str(row.get("method", "")))
            rows.append(row)
    return pd.DataFrame(rows)


def _metric_pivot(df: pd.DataFrame, metric: str, methods: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["method"])
    table = df.groupby(["method", "benchmark"], dropna=False)[metric].mean().unstack("benchmark").reset_index()
    return _sort_methods(table, methods)


def _with_invalid_indicator(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pred_norm"] = out["pred"].map(_normalize_label) if "pred" in out.columns else "invalid"
    out["invalid_rate"] = (~out["pred_norm"].isin(VALID_LABELS)).astype(float)
    return out


def _aggregate_metrics(df: pd.DataFrame, group_cols: list[str], metrics: list[str], methods: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + metrics)
    cols = [metric for metric in metrics if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric])]
    if not cols:
        return pd.DataFrame(columns=group_cols + metrics)
    filtered = df[df["method"].isin(methods)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=group_cols + cols)
    summary = filtered.groupby(group_cols, dropna=False)[cols].mean().reset_index()
    return _sort_methods(summary, methods)


def _diagnostic_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for _, pred in predictions[predictions["method"].isin(TRACE_METHODS)].iterrows():
        trace = _decode_trace(pred.get("raw_response"))
        diagnostics = trace.get("diagnostics") if isinstance(trace, dict) else None
        if not isinstance(diagnostics, list):
            continue
        for i, diag in enumerate(diagnostics):
            if not isinstance(diag, dict):
                continue
            status = str(diag.get("failure_type") or diag.get("status") or "unknown")
            rows.append(
                {
                    "benchmark": pred["benchmark"],
                    "method": pred["method"],
                    "example_id": pred["example_id"],
                    "probe_id": i,
                    "probe_type": diag.get("probe_type"),
                    "cf_family": diag.get("cf_family"),
                    "expected_relation": diag.get("expected_relation"),
                    "diagnostic_status": status,
                    "is_ok": float(status == "ok"),
                    "is_failed": float(status != "ok"),
                    "is_invalid": float(status in {"invalid", "invalid_probe"}),
                }
            )
    return pd.DataFrame(rows)


def _decode_trace(value) -> dict:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _has_repair(trace: dict, method: str) -> float:
    if method in {"llm_cage", "llm_cage_select"}:
        return float(trace.get("repaired_raw") is not None or trace.get("repair_raw") is not None)
    return float(trace.get("repair_raw") is not None)


def _normalize_label(value) -> str:
    if pd.isna(value):
        return "invalid"
    if isinstance(value, bool):
        return "true" if value else "false"
    normalized = str(value).strip().lower().strip("`*_ .:;!\"'")
    aliases = {
        "yes": "true",
        "entailed": "true",
        "entailment": "true",
        "valid": "true",
        "no": "false",
        "contradiction": "false",
        "contradicts": "false",
        "neutral": "unknown",
        "uncertain": "unknown",
        "inconclusive": "unknown",
        "unsupported": "unknown",
    }
    return aliases.get(normalized, normalized if normalized in VALID_LABELS else "invalid")


def _sort_methods(df: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    if df.empty or "method" not in df.columns:
        return df
    order = {method: i for i, method in enumerate(methods)}
    out = df.copy()
    out["_method_order"] = out["method"].map(order).fillna(len(order))
    sort_cols = [col for col in ["benchmark", "_method_order", "method", "cf_family", "diagnostic_status"] if col in out.columns]
    return out.sort_values(sort_cols).drop(columns="_method_order").reset_index(drop=True)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _mean_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else float("nan")


def _write_percent_latex(df: pd.DataFrame, path: Path, value_cols: list[str]) -> None:
    if df.empty:
        path.write_text("", encoding="utf-8")
        return
    out = df.copy()
    for col in value_cols:
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{100 * v:.1f}")
    out.to_latex(path, index=False, escape=True)


def _grouped_bars(ax, df: pd.DataFrame, benchmarks: list[str], methods: tuple[str, ...], metric: str) -> None:
    width = min(0.78 / max(len(methods), 1), 0.14)
    offsets = [(i - (len(methods) - 1) / 2) * width for i in range(len(methods))]
    for offset, method in zip(offsets, methods):
        values = []
        for benchmark in benchmarks:
            subset = df[(df["benchmark"] == benchmark) & (df["method"] == method)]
            values.append(float(subset[metric].mean()) if not subset.empty else 0.0)
        xs = [i + offset for i in range(len(benchmarks))]
        ax.bar(xs, values, width=width * 0.92, color=METHOD_COLORS.get(method, "#898781"), label=METHOD_LABELS.get(method, method))


def _finish_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, output_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=240)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    specs = args.run + args.runs
    build_paper_outputs(specs, args.output_dir, make_figures=not args.no_figures)
    out = Path(args.output_dir)
    print(f"Wrote paper tables and figures to {out.as_posix()}.")


if __name__ == "__main__":
    main()
