from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .baselines import offline_predictions
from .counterfactuals import generate_counterfactuals
from .datasets import load_folio_examples, load_prontoqa_examples, load_proofwriter_examples, write_parse_failures
from .evaluate import aggregate, aggregate_research, plot_metrics, summarize_base, summarize_counterfactual, write_error_analysis
from .generator import generate_examples, write_jsonl
from .llm_client import run_llm_if_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run counterfactual logical reasoning experiments.")
    parser.add_argument("--offline", action="store_true", help="Run deterministic offline baselines.")
    parser.add_argument("--llm", action="store_true", help="Run optional LLM prompts when credentials are available.")
    parser.add_argument("--llm-baseline-methods", default="all", help="Comma-separated older LLM baselines to run: logiclm,symbcot,vericot,symbcot_cage, all, or none.")
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--dataset", choices=["auto", "proofwriter", "prontoqa", "folio", "synthetic"], default="auto")
    parser.add_argument("--proofwriter-path", default=None, help="Path to a local ProofWriter JSON/JSONL file or directory.")
    parser.add_argument("--proofwriter-root", default=None, help="Directory containing data-train.jsonl, data-dev.jsonl, and data-test.jsonl.")
    parser.add_argument("--prontoqa-path", default="data/raw/prontoqa2/ProntoQA_dev_gpt-4.json", help="Path to PrOntoQA JSON, ZIP, or directory.")
    parser.add_argument("--folio-path", default="data/raw/FOLIO/data/v0.0/folio-validation.jsonl", help="Path to FOLIO JSONL file.")
    parser.add_argument("--splits", default="train,dev,test", help="Comma-separated ProofWriter splits to run when --proofwriter-root is used.")
    parser.add_argument("--max-per-split", type=int, default=None, help="Maximum parseable ProofWriter examples per split.")
    parser.add_argument("--counterfactuals-per-example", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--data-dir", default="data/generated")
    return parser.parse_args()


def _split_names(args: argparse.Namespace) -> list[str]:
    return [split.strip() for split in args.splits.split(",") if split.strip()]


def _proofwriter_file(root: str | Path, split: str) -> Path:
    aliases = {"validation": "dev", "valid": "dev"}
    split = aliases.get(split, split)
    root = Path(root)
    candidates = [root / f"data-{split}.jsonl", root / f"meta-{split}.jsonl"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_split_examples(args: argparse.Namespace) -> tuple[dict[str, list], list[dict]]:
    limit = args.max_per_split if args.max_per_split is not None else args.n_examples
    split_examples: dict[str, list] = {}
    reports: list[dict] = []
    for split in _split_names(args):
        path = _proofwriter_file(args.proofwriter_root, split)
        examples, report = load_proofwriter_examples(path, limit=limit, split=split)
        split_examples[split] = examples
        reports.append(report.to_dict(split=split))
        print(
            f"Loaded {report.parsed} parseable ProofWriter {split} examples from {report.loaded} rows "
            f"({report.skipped} skipped, coverage={report.parsed / report.loaded if report.loaded else 0:.3f})."
        )
    return split_examples, reports


def _load_base_examples(args: argparse.Namespace):
    if args.dataset == "folio":
        examples = load_folio_examples(args.folio_path, limit=args.n_examples, split="folio")
        if not examples:
            raise RuntimeError("No FOLIO examples found. Check --folio-path.")
        raise RuntimeError("FOLIO is a FOL natural-language benchmark; run it with --llm instead of --offline.")
    if args.dataset == "prontoqa":
        examples, report = load_prontoqa_examples(args.prontoqa_path, limit=args.n_examples, split="prontoqa")
        if not examples:
            raise RuntimeError("No parseable PrOntoQA examples found. Check --prontoqa-path.")
        print(
            f"Loaded {report.parsed} parseable PrOntoQA examples from {report.loaded} rows "
            f"({report.skipped} skipped)."
        )
        return examples, "prontoqa", report
    if args.dataset in {"auto", "proofwriter"} and args.proofwriter_path:
        examples, report = load_proofwriter_examples(args.proofwriter_path, limit=args.n_examples, split="proofwriter")
        if examples:
            print(
                f"Loaded {report.parsed} parseable ProofWriter examples from {report.loaded} rows "
                f"({report.skipped} skipped)."
            )
            return examples, "proofwriter_cf", report
        if args.dataset == "proofwriter":
            raise RuntimeError("No parseable ProofWriter examples found. Check the input format or use --dataset synthetic.")
        print("No parseable ProofWriter examples found; falling back to synthetic data.")
    elif args.dataset == "proofwriter":
        raise RuntimeError("--dataset proofwriter requires --proofwriter-path or --proofwriter-root.")
    examples = generate_examples(args.n_examples, seed=args.seed, max_depth=args.max_depth)
    for ex in examples:
        ex.split = "synthetic"
    return examples, "synthetic_control", None


def run_offline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.proofwriter_root:
        run_split_offline(args, data_dir, output_dir, figures_dir)
        return

    examples, dataset_name, report = _load_base_examples(args)
    cfs = generate_counterfactuals(examples, seed=args.seed, max_per_example=args.counterfactuals_per_example)
    write_jsonl(data_dir / f"{dataset_name}_base.jsonl", examples)
    write_jsonl(data_dir / f"{dataset_name}_counterfactual.jsonl", cfs)
    if report is not None:
        pd.DataFrame([report.to_dict(split="proofwriter")]).to_csv(output_dir / "parse_report_by_split.csv", index=False)
        write_parse_failures(output_dir / "parse_failures_proofwriter.csv", report, split="proofwriter")

    base_preds = offline_predictions(examples, train_examples=examples)
    cf_preds = offline_predictions(cfs, train_examples=examples)

    base_df = summarize_base(examples, base_preds)
    cf_df = summarize_counterfactual(examples, cfs, cf_preds)
    base_df.to_csv(output_dir / "results_base_offline.csv", index=False)
    cf_df.to_csv(output_dir / "results_counterfactual_offline.csv", index=False)
    aggregate(base_df, ["split", "method"]).to_csv(output_dir / "summary_base_offline.csv", index=False)
    aggregate(cf_df, ["split", "method"]).to_csv(output_dir / "summary_counterfactual_offline.csv", index=False)
    aggregate_research(base_df, cf_df).to_csv(output_dir / "summary_offline_by_split.csv", index=False)
    write_grouped_summaries(base_df, cf_df, output_dir)
    write_diagnostic_bundle_stats(cf_df, output_dir)
    plot_metrics(base_df, cf_df, figures_dir)
    write_error_analysis(base_df, cf_df, output_dir)

    print(f"Dataset: {dataset_name}.")
    print(f"Generated {len(examples)} base examples and {len(cfs)} counterfactual examples.")
    print(f"Wrote outputs to {output_dir}.")


def write_grouped_summaries(base_df: pd.DataFrame, cf_df: pd.DataFrame, output_dir: Path) -> None:
    specs = {
        "summary_offline_by_label.csv": ["split", "method", "gold"],
        "summary_offline_by_depth.csv": ["split", "method", "depth"],
        "summary_offline_by_intervention.csv": ["split", "method", "intervention_type"],
        "summary_offline_by_intervention_type.csv": ["split", "method", "intervention_type"],
        "summary_offline_by_diagnostic_dimension.csv": ["split", "method", "diagnostic_dimension"],
        "summary_offline_by_relation.csv": ["split", "method", "expected_relation"],
        "summary_offline_by_cf_family.csv": ["split", "method", "cf_family"],
        "summary_offline_by_proof_relation.csv": ["split", "method", "proof_relation"],
        "summary_offline_by_support_overlap.csv": ["split", "method", "support_overlap"],
    }
    for filename, group_cols in specs.items():
        aggregate_research(base_df, cf_df, group_cols=group_cols).to_csv(output_dir / filename, index=False)


def write_diagnostic_bundle_stats(cf_df: pd.DataFrame, output_dir: Path) -> None:
    if cf_df.empty or "cf_family" not in cf_df.columns:
        pd.DataFrame().to_csv(output_dir / "diagnostic_bundle_stats.csv", index=False)
        return
    unique = cf_df.drop_duplicates(["split", "source_id", "example_id"])
    rows = []
    for split, group in unique.groupby("split", dropna=False):
        row = {
            "split": split,
            "num_source_examples": int(group["source_id"].nunique()),
            "num_bundles": int(group["bundle_id"].nunique()) if "bundle_id" in group.columns else int(group["source_id"].nunique()),
            "num_counterfactuals": int(len(group)),
            "avg_variants_per_source": float(group.groupby("source_id")["example_id"].nunique().mean()),
        }
        for family, count in group["cf_family"].value_counts().sort_index().items():
            row[f"num_{family}"] = int(count)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "diagnostic_bundle_stats.csv", index=False)


def run_split_offline(args: argparse.Namespace, data_dir: Path, output_dir: Path, figures_dir: Path) -> None:
    split_examples, reports = _load_split_examples(args)
    train_examples = split_examples.get("train") or next(iter(split_examples.values()))
    parse_df = pd.DataFrame(reports)
    parse_df.to_csv(output_dir / "parse_report_by_split.csv", index=False)

    all_examples = []
    all_cfs = []
    base_frames = []
    cf_frames = []
    for split, examples in split_examples.items():
        cfs = generate_counterfactuals(examples, seed=args.seed, max_per_example=args.counterfactuals_per_example)
        all_examples.extend(examples)
        all_cfs.extend(cfs)
        write_jsonl(data_dir / f"proofwriter_{split}_base.jsonl", examples)
        write_jsonl(data_dir / f"proofwriter_{split}_counterfactual.jsonl", cfs)

        base_preds = offline_predictions(examples, train_examples=train_examples)
        cf_preds = offline_predictions(cfs, train_examples=train_examples)
        base_frames.append(summarize_base(examples, base_preds))
        cf_frames.append(summarize_counterfactual(examples, cfs, cf_preds))

        _, report = load_proofwriter_examples(_proofwriter_file(args.proofwriter_root, split), limit=args.max_per_split or args.n_examples, split=split)
        write_parse_failures(output_dir / f"parse_failures_{split}.csv", report, split=split)

    base_df = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()
    cf_df = pd.concat(cf_frames, ignore_index=True) if cf_frames else pd.DataFrame()
    base_df.to_csv(output_dir / "results_base_offline_by_split.csv", index=False)
    cf_df.to_csv(output_dir / "results_counterfactual_offline_by_split.csv", index=False)
    aggregate(base_df, ["split", "method"]).to_csv(output_dir / "summary_base_offline.csv", index=False)
    aggregate(cf_df, ["split", "method"]).to_csv(output_dir / "summary_counterfactual_offline.csv", index=False)
    aggregate_research(base_df, cf_df).to_csv(output_dir / "summary_offline_by_split.csv", index=False)
    write_grouped_summaries(base_df, cf_df, output_dir)
    plot_metrics(base_df, cf_df, figures_dir, parse_df=parse_df)
    write_error_analysis(base_df, cf_df, output_dir)

    print(f"Dataset: proofwriter_cf_split.")
    print(f"Generated {len(all_examples)} base examples and {len(all_cfs)} counterfactual examples across {len(split_examples)} splits.")
    print(f"Wrote outputs to {output_dir}.")


def main() -> None:
    args = parse_args()
    if not args.offline and not args.llm:
        args.offline = True
    if args.offline:
        run_offline(args)
    if args.llm:
        run_llm_if_available(args)


if __name__ == "__main__":
    main()
