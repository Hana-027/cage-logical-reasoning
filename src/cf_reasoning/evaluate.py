from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import pandas as pd

from .schema import CounterfactualExample, Example, Prediction

LABELS = ("true", "false", "unknown")
PLOT_METHODS = ("train_majority", "depth_majority", "fact_lookup", "one_step_rule", "lexical_overlap", "mpss", "symbolic_oracle")
METHOD_LABELS = {
    "train_majority": "Train majority",
    "depth_majority": "Depth majority",
    "fact_lookup": "Fact lookup",
    "one_step_rule": "One-step rule",
    "lexical_overlap": "Lexical overlap",
    "mpss": "MPSS",
    "symbolic_oracle": "Symbolic oracle",
}
METHOD_COLORS = {
    "train_majority": "#9CA3AF",
    "depth_majority": "#7C8DA6",
    "fact_lookup": "#4C78A8",
    "one_step_rule": "#72B7B2",
    "lexical_overlap": "#F2A541",
    "mpss": "#D62728",
    "symbolic_oracle": "#111111",
}


def _style_for_method(method: str) -> dict[str, object]:
    if method == "mpss":
        return {"color": METHOD_COLORS[method], "linewidth": 2.2, "marker": "o", "markersize": 5.5, "zorder": 5}
    if method == "symbolic_oracle":
        return {"color": METHOD_COLORS[method], "linewidth": 2.0, "linestyle": "--", "marker": "s", "markersize": 4.8, "zorder": 4}
    return {"color": METHOD_COLORS.get(method, "#6B7280"), "linewidth": 1.3, "marker": "o", "markersize": 3.8, "alpha": 0.8, "zorder": 2}


def _ordered_methods(values: list[str] | pd.Series) -> list[str]:
    present = set(values)
    return [method for method in PLOT_METHODS if method in present]


def _finish_axis(ax, title: str, ylabel: str, xlabel: str = "") -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def support_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right) if left | right else 0.0


def exact_match(pred: str, gold: str) -> int:
    return int(pred == gold)


def f1_score(pred_ids: set[str], gold_ids: set[str]) -> tuple[float, float, float]:
    if not pred_ids and not gold_ids:
        return 1.0, 1.0, 1.0
    if not pred_ids:
        return 0.0, 0.0, 0.0
    tp = len(pred_ids & gold_ids)
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall = tp / len(gold_ids) if gold_ids else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _attribution_consistency(proof_relation: str, pred_support: set[str], source_support: set[str], target_support: set[str]) -> int | None:
    if proof_relation == "broken":
        return int(bool(pred_support & target_support)) if target_support else int(not pred_support)
    if proof_relation == "preserved_same_support":
        return int(support_jaccard(pred_support, target_support) >= 0.5)
    if proof_relation in {"shifted_support", "preserved_new_support"}:
        return int(support_jaccard(pred_support, target_support) > support_jaccard(pred_support, source_support))
    return None


def summarize_base(examples: list[Example], predictions: list[Prediction]) -> pd.DataFrame:
    by_id = {ex.id: ex for ex in examples}
    rows = []
    for pred in predictions:
        ex = by_id[pred.example_id]
        precision, recall, f1 = f1_score(set(pred.premise_ids), set(ex.support_ids))
        row = {
            "split": ex.split,
            "example_id": pred.example_id,
            "method": pred.method,
            "gold": ex.label,
            "pred": pred.label,
            "depth": ex.depth,
            "accuracy": exact_match(pred.label, ex.label),
            "attr_precision": precision,
            "attr_recall": recall,
            "attr_f1": f1,
            "exact_support_match": int(set(pred.premise_ids) == set(ex.support_ids)),
            "support_ids": " ".join(ex.support_ids),
            "pred_support_ids": " ".join(pred.premise_ids),
            "text": ex.text,
        }
        row.update(_prediction_trace_fields(pred))
        rows.append(row)
    return pd.DataFrame(rows)


def _prediction_trace_fields(pred: Prediction) -> dict[str, object]:
    if pred.method != "mpss" or not pred.raw_response:
        return {}
    try:
        data = json.loads(pred.raw_response)
    except json.JSONDecodeError:
        return {}
    fields: dict[str, object] = {}
    for key in ["strategy", "expansions", "positive_found", "negative_found", "contradiction"]:
        if key in data:
            fields[f"mpss_{key}"] = data[key]
    return fields

def summarize_counterfactual(
    base_examples: list[Example],
    cf_examples: list[CounterfactualExample],
    predictions: list[Prediction],
) -> pd.DataFrame:
    source_labels = {ex.id: ex.label for ex in base_examples}
    by_id = {ex.id: ex for ex in cf_examples}
    rows = []
    for pred in predictions:
        ex = by_id[pred.example_id]
        expected_changed = ex.expected_relation == "changed"
        pred_changed = pred.label != source_labels[ex.source_id]
        pred_support = set(pred.premise_ids)
        target_support = set(ex.target_support_ids or ex.support_ids)
        source_support = set(ex.source_support_ids)
        precision, recall, f1 = f1_score(pred_support, target_support)
        pred_target_jaccard = support_jaccard(pred_support, target_support)
        pred_source_jaccard = support_jaccard(pred_support, source_support)
        shifted = ex.proof_relation in {"shifted_support", "preserved_new_support"}
        support_shift_detected = int(pred_target_jaccard > pred_source_jaccard) if shifted else None
        attribution_consistency = _attribution_consistency(ex.proof_relation, pred_support, source_support, target_support)
        row = {
            "split": ex.split,
            "example_id": pred.example_id,
            "source_id": ex.source_id,
            "bundle_id": ex.bundle_id,
            "parent_id": ex.parent_id,
            "method": pred.method,
            "intervention_type": ex.intervention_type,
            "cf_family": ex.cf_family,
            "diagnostic_dimension": ex.diagnostic_dimension,
            "proof_relation": ex.proof_relation,
            "expected_relation": ex.expected_relation,
            "changed_ids": " ".join(ex.changed_ids),
            "source_support_ids": " ".join(ex.source_support_ids),
            "target_support_ids": " ".join(ex.target_support_ids or ex.support_ids),
            "removed_support_ids": " ".join(ex.removed_support_ids),
            "added_support_ids": " ".join(ex.added_support_ids),
            "support_overlap": ex.support_overlap,
            "edit_distance": ex.edit_distance,
            "is_minimal": ex.is_minimal,
            "conflict_label": ex.conflict_label,
            "gold": ex.label,
            "pred": pred.label,
            "source_gold": source_labels[ex.source_id],
            "depth": ex.depth,
            "accuracy": exact_match(pred.label, ex.label),
            "counterfactual_consistency": int(pred_changed == expected_changed),
            "label_change_accuracy": int(pred.label == ex.label) if expected_changed else None,
            "irrelevant_robustness": int(pred.label == ex.label) if not expected_changed else None,
            "attr_precision": precision,
            "attr_recall": recall,
            "attr_f1": f1,
            "source_target_support_jaccard": support_jaccard(source_support, target_support),
            "pred_target_support_jaccard": pred_target_jaccard,
            "pred_source_support_jaccard": pred_source_jaccard,
            "support_shift_detected": support_shift_detected,
            "attribution_consistency": attribution_consistency,
            "proof_break_sensitivity": int(pred.label == ex.label) if ex.cf_family == "proof_breaking" else None,
            "proof_preserve_robustness": int(pred.label == ex.label) if ex.cf_family == "proof_preserving" else None,
            "alternate_proof_awareness": int(pred.label == ex.label and pred_target_jaccard > pred_source_jaccard) if ex.cf_family in {"support_shift", "alternate_proof"} else None,
            "contradiction_detection": int(pred.label in {"unknown", "ambiguous"}) if ex.cf_family == "contradiction_injection" else None,
            "distractor_robustness": int(pred.label == ex.label) if ex.cf_family == "distractor_injection" else None,
            "entity_binding_consistency": int(pred.label == ex.label) if ex.cf_family == "entity_swap" else None,
            "rule_structure_sensitivity": int(pred.label == ex.label) if ex.cf_family == "rule_structure_intervention" else None,
            "paraphrase_consistency": int(pred.label == ex.label) if ex.cf_family == "paraphrase_preserving" else None,
            "minimal_edit_sensitivity": int(pred.label == ex.label) if ex.is_minimal else None,
            "support_ids": " ".join(ex.support_ids),
            "pred_support_ids": " ".join(pred.premise_ids),
            "text": ex.text,
        }
        row.update(_prediction_trace_fields(pred))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = group_cols or ["method"]
    numeric_cols = [
        c
        for c in df.columns
        if c
        not in {
            "example_id",
            "source_id",
            "bundle_id",
            "parent_id",
            "gold",
            "pred",
            "source_gold",
            "intervention_type",
            "cf_family",
            "diagnostic_dimension",
            "proof_relation",
            "expected_relation",
            "method",
            "split",
            "support_ids",
            "source_support_ids",
            "target_support_ids",
            "removed_support_ids",
            "added_support_ids",
            "pred_support_ids",
            "text",
            "changed_ids",
            "conflict_label",
        }
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()


def aggregate_research(base_df: pd.DataFrame, cf_df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or ["split", "method"]
    rows = []
    for df, scope in [(base_df, "base"), (cf_df, "counterfactual")]:
        if df.empty or any(col not in df.columns for col in group_cols):
            continue
        for group_key, group in df.groupby(group_cols, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = {"scope": scope, "n": len(group)}
            row.update(dict(zip(group_cols, group_key)))
            for metric in _metric_columns(group):
                values = group[metric].dropna()
                if values.empty:
                    continue
                mean = float(values.mean())
                lo, hi = bootstrap_ci(group, metric, id_col="source_id" if "source_id" in group.columns else "example_id")
                row[metric] = mean
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
            row["macro_f1"] = macro_f1(group)
            row["balanced_accuracy"] = balanced_accuracy(group)
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(df: pd.DataFrame, metric: str, id_col: str = "example_id", n_boot: int = 200, seed: int = 42) -> tuple[float, float]:
    values = df[[id_col, metric]].dropna()
    if values.empty:
        return 0.0, 0.0
    by_id = values.groupby(id_col)[metric].mean()
    ids = list(by_id.index)
    if len(ids) <= 1:
        mean = float(by_id.mean())
        return mean, mean
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        selected = [rng.choice(ids) for _ in ids]
        samples.append(float(by_id.loc[selected].mean()))
    samples.sort()
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]


def macro_f1(df: pd.DataFrame) -> float:
    labels = tuple(label for label in LABELS if ((df["gold"] == label) | (df["pred"] == label)).any())
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = int(((df["gold"] == label) & (df["pred"] == label)).sum())
        fp = int(((df["gold"] != label) & (df["pred"] == label)).sum())
        fn = int(((df["gold"] == label) & (df["pred"] != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores))


def balanced_accuracy(df: pd.DataFrame) -> float:
    recalls = []
    for label in LABELS:
        subset = df[df["gold"] == label]
        if not subset.empty:
            recalls.append(float((subset["pred"] == label).mean()))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def write_error_analysis(base_df: pd.DataFrame, cf_df: pd.DataFrame, output_dir: str | Path, limit: int = 100) -> None:
    output_dir = Path(output_dir) / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_failure_sample(base_df[base_df["accuracy"] == 0], output_dir / "base_failures.csv", limit)
    _write_failure_sample(cf_df[(cf_df["expected_relation"] == "changed") & (cf_df["accuracy"] == 0)], output_dir / "counterfactual_changed_failures.csv", limit)
    _write_failure_sample(cf_df[(cf_df["expected_relation"] == "preserved") & (cf_df["accuracy"] == 0)], output_dir / "counterfactual_preserved_failures.csv", limit)
    _write_failure_sample(pd.concat([base_df, cf_df], ignore_index=True).query("attr_f1 < 1"), output_dir / "attribution_failures.csv", limit)


def plot_metrics(base_df: pd.DataFrame, cf_df: pd.DataFrame, output_dir: str | Path, parse_df: pd.DataFrame | None = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if parse_df is not None and not parse_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(parse_df["split"], parse_df["coverage"])
        ax.set_title("ProofWriter parse coverage by split")
        ax.set_xlabel("Split")
        ax.set_ylabel("Parse coverage")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(output_dir / "parse_coverage_by_split.png", dpi=180)
        plt.close(fig)

    if not base_df.empty:
        by_depth = base_df.groupby(["method", "depth"])["accuracy"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method in _ordered_methods(by_depth["method"]):
            group = by_depth[by_depth["method"] == method].sort_values("depth")
            ax.plot(group["depth"], group["accuracy"], label=METHOD_LABELS.get(method, method), **_style_for_method(method))
        _finish_axis(ax, "Original accuracy by reasoning depth", "Accuracy", "Reasoning depth")
        ax.legend(fontsize=8, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "accuracy_by_depth.png", dpi=220)
        plt.close(fig)

        label_dist = base_df.drop_duplicates("example_id").groupby(["split", "gold"]).size().reset_index(name="count")
        pivot = label_dist.pivot(index="split", columns="gold", values="count").fillna(0)
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Gold label distribution by split")
        ax.set_xlabel("Split")
        ax.set_ylabel("Examples")
        fig.tight_layout()
        fig.savefig(output_dir / "label_distribution_by_split.png", dpi=180)
        plt.close(fig)

        mpss_rows = base_df[(base_df["method"] == "mpss") & base_df["mpss_strategy"].notna()] if "mpss_strategy" in base_df.columns else pd.DataFrame()
        if not mpss_rows.empty:
            strategy_counts = mpss_rows["mpss_strategy"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            strategy_counts.plot(kind="bar", ax=ax)
            ax.set_title("MPSS proof strategy distribution")
            ax.set_xlabel("Selected strategy")
            ax.set_ylabel("Examples")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / "mpss_strategy_distribution.png", dpi=180)
            plt.close(fig)

    if not cf_df.empty:
        ccr = cf_df.groupby(["method", "intervention_type"])["counterfactual_consistency"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        for method in _ordered_methods(ccr["method"]):
            group = ccr[ccr["method"] == method].sort_values("intervention_type")
            ax.plot(group["intervention_type"], group["counterfactual_consistency"], label=METHOD_LABELS.get(method, method), **_style_for_method(method))
        _finish_axis(ax, "Counterfactual consistency by intervention", "CCR")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.legend(fontsize=8, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "counterfactual_consistency.png", dpi=220)
        plt.close(fig)

        relation = cf_df.groupby(["method", "expected_relation"])["accuracy"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for method in _ordered_methods(relation["method"]):
            group = relation[relation["method"] == method].sort_values("expected_relation")
            ax.plot(group["expected_relation"], group["accuracy"], label=METHOD_LABELS.get(method, method), **_style_for_method(method))
        _finish_axis(ax, "Changed vs. preserved counterfactual accuracy", "Accuracy")
        ax.legend(fontsize=8, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "changed_vs_preserved_accuracy.png", dpi=220)
        plt.close(fig)

        f1 = cf_df.groupby(["method", "depth"])["attr_f1"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method in _ordered_methods(f1["method"]):
            group = f1[f1["method"] == method].sort_values("depth")
            ax.plot(group["depth"], group["attr_f1"], label=METHOD_LABELS.get(method, method), **_style_for_method(method))
        _finish_axis(ax, "Attribution F1 by reasoning depth", "Attribution F1", "Reasoning depth")
        ax.legend(fontsize=8, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "attribution_f1_by_depth.png", dpi=220)
        plt.close(fig)

        if "cf_family" in cf_df.columns:
            family = cf_df.groupby(["method", "cf_family"])["accuracy"].mean().reset_index()
            pivot = family.pivot(index="cf_family", columns="method", values="accuracy").fillna(0)
            cols = [col for col in PLOT_METHODS if col in pivot.columns]
            if cols:
                fig, ax = plt.subplots(figsize=(9.2, 4.6))
                pivot[cols].plot(kind="bar", ax=ax, color=[METHOD_COLORS[col] for col in cols], width=0.78)
                _finish_axis(ax, "Accuracy by proof-level counterfactual family", "Accuracy", "Counterfactual family")
                ax.tick_params(axis="x", rotation=30, labelsize=8)
                ax.legend([METHOD_LABELS.get(col, col) for col in cols], fontsize=8, ncol=2, frameon=False)
                fig.tight_layout()
                fig.savefig(output_dir / "proof_family_accuracy.png", dpi=220)
                plt.close(fig)

            if "diagnostic_dimension" in cf_df.columns:
                dimension = cf_df.groupby(["method", "diagnostic_dimension"])["accuracy"].mean().reset_index()
                pivot = dimension.pivot(index="diagnostic_dimension", columns="method", values="accuracy").fillna(0)
                cols = [col for col in PLOT_METHODS if col in pivot.columns]
                if cols:
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    pivot[cols].plot(kind="bar", ax=ax)
                    ax.set_title("Accuracy by diagnostic dimension")
                    ax.set_xlabel("Diagnostic dimension")
                    ax.set_ylabel("Accuracy")
                    ax.set_ylim(0, 1.05)
                    fig.tight_layout()
                    fig.savefig(output_dir / "diagnostic_dimension_accuracy.png", dpi=180)
                    plt.close(fig)

            if "attribution_consistency" in cf_df.columns:
                attr = cf_df.groupby(["method", "cf_family"])["attribution_consistency"].mean().reset_index()
                pivot = attr.pivot(index="cf_family", columns="method", values="attribution_consistency").fillna(0)
                cols = [col for col in PLOT_METHODS if col in pivot.columns]
                if cols:
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    pivot[cols].plot(kind="bar", ax=ax)
                    ax.set_title("Attribution consistency by proof-level family")
                    ax.set_xlabel("Counterfactual family")
                    ax.set_ylabel("Attribution consistency")
                    ax.set_ylim(0, 1.05)
                    fig.tight_layout()
                    fig.savefig(output_dir / "attribution_consistency_by_family.png", dpi=180)
                    plt.close(fig)

            if "support_shift_detected" in cf_df.columns:
                shift = cf_df[cf_df["support_shift_detected"].notna()]
                if not shift.empty:
                    shift_summary = shift.groupby(["method", "cf_family"])["support_shift_detected"].mean().reset_index()
                    pivot = shift_summary.pivot(index="cf_family", columns="method", values="support_shift_detected").fillna(0)
                    cols = [col for col in PLOT_METHODS if col in pivot.columns]
                    if cols:
                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        pivot[cols].plot(kind="bar", ax=ax)
                        ax.set_title("Support-shift awareness")
                        ax.set_xlabel("Counterfactual family")
                        ax.set_ylabel("Support shift detected")
                        ax.set_ylim(0, 1.05)
                        fig.tight_layout()
                        fig.savefig(output_dir / "support_shift_awareness.png", dpi=180)
                        plt.close(fig)


def _metric_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "accuracy",
        "counterfactual_consistency",
        "label_change_accuracy",
        "irrelevant_robustness",
        "attr_precision",
        "attr_recall",
        "attr_f1",
        "exact_support_match",
        "source_target_support_jaccard",
        "pred_target_support_jaccard",
        "pred_source_support_jaccard",
        "support_shift_detected",
        "attribution_consistency",
        "proof_break_sensitivity",
        "proof_preserve_robustness",
        "alternate_proof_awareness",
        "contradiction_detection",
        "distractor_robustness",
        "entity_binding_consistency",
        "rule_structure_sensitivity",
        "paraphrase_consistency",
        "minimal_edit_sensitivity",
    ]
    return [col for col in preferred if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def _write_failure_sample(df: pd.DataFrame, path: Path, limit: int) -> None:
    cols = [
        "split",
        "method",
        "example_id",
        "source_id",
        "bundle_id",
        "parent_id",
        "intervention_type",
        "cf_family",
        "diagnostic_dimension",
        "proof_relation",
        "expected_relation",
        "changed_ids",
        "source_support_ids",
        "target_support_ids",
        "removed_support_ids",
        "added_support_ids",
        "support_overlap",
        "edit_distance",
        "is_minimal",
        "conflict_label",
        "support_ids",
        "pred_support_ids",
        "gold",
        "mpss_strategy",
        "mpss_expansions",
        "mpss_positive_found",
        "mpss_negative_found",
        "mpss_contradiction",
        "text",
    ]
    existing = [col for col in cols if col in df.columns]
    df[existing].head(limit).to_csv(path, index=False)
