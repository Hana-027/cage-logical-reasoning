from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .datasets import load_folio_examples, load_prontoqa_examples, load_proofwriter_examples
from .fair_eval import REQUIRED_LLM_METHODS
from .llm_baselines import normalize_baseline_methods

DEFAULT_PROOFWRITER_ROOT = "data/raw/proofwriter"
DEFAULT_PRONTOQA_PATH = "data/raw/prontoqa2/ProntoQA_dev_gpt-4.json"
DEFAULT_FOLIO_PATH = "data/raw/FOLIO/data/v0.0/folio-validation.jsonl"


def build_preflight_report(
    proofwriter_root: str | Path = DEFAULT_PROOFWRITER_ROOT,
    prontoqa_path: str | Path = DEFAULT_PRONTOQA_PATH,
    folio_path: str | Path = DEFAULT_FOLIO_PATH,
    n_examples: int = 20,
) -> dict[str, Any]:
    report = {
        "method_set": {
            "baseline_methods_all": list(normalize_baseline_methods("all")),
            "required_result_methods": list(REQUIRED_LLM_METHODS),
        },
        "datasets": {},
    }
    report["datasets"]["proofwriter"] = _proofwriter_summary(Path(proofwriter_root), n_examples)
    report["datasets"]["prontoqa"] = _prontoqa_summary(Path(prontoqa_path), n_examples)
    report["datasets"]["folio"] = _folio_summary(Path(folio_path), n_examples)
    report["fairness_ready"] = _fairness_ready(report)
    return report


def write_preflight_report(report: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for benchmark, item in report["datasets"].items():
        rows.append(
            {
                "benchmark": benchmark,
                "path": item.get("path", ""),
                "loaded": item.get("loaded", item.get("count", 0)),
                "parsed": item.get("parsed", item.get("count", 0)),
                "skipped": item.get("skipped", 0),
                "coverage": item.get("coverage", 1.0),
                "label_counts": json.dumps(item.get("label_counts", {}), ensure_ascii=False, sort_keys=True),
                "first_ids": ",".join(item.get("first_ids", [])),
                "status": item.get("status", "ok"),
            }
        )
    pd.DataFrame(rows).to_csv(output / "preflight_datasets.csv", index=False)


def _proofwriter_summary(root: Path, n_examples: int) -> dict[str, Any]:
    path = _proofwriter_file(root, "test")
    examples, load_report = load_proofwriter_examples(path, limit=n_examples, split="test")
    return {
        "status": "ok" if examples else "empty",
        "path": str(path),
        "loaded": load_report.loaded,
        "parsed": load_report.parsed,
        "skipped": load_report.skipped,
        "coverage": load_report.parsed / load_report.loaded if load_report.loaded else 0.0,
        "label_counts": load_report.label_counts,
        "depth_counts": load_report.depth_counts,
        "first_ids": [ex.id for ex in examples[:5]],
    }


def _prontoqa_summary(path: Path, n_examples: int) -> dict[str, Any]:
    examples, load_report = load_prontoqa_examples(path, limit=n_examples, split="prontoqa")
    return {
        "status": "ok" if examples else "empty",
        "path": str(path),
        "loaded": load_report.loaded,
        "parsed": load_report.parsed,
        "skipped": load_report.skipped,
        "coverage": load_report.parsed / load_report.loaded if load_report.loaded else 0.0,
        "label_counts": load_report.label_counts,
        "depth_counts": load_report.depth_counts,
        "first_ids": [ex.id for ex in examples[:5]],
    }


def _folio_summary(path: Path, n_examples: int) -> dict[str, Any]:
    examples = load_folio_examples(path, limit=n_examples, split="folio")
    labels = Counter(ex.label for ex in examples)
    return {
        "status": "ok" if examples else "empty",
        "path": str(path),
        "count": len(examples),
        "coverage": 1.0 if examples else 0.0,
        "label_counts": dict(labels),
        "first_ids": [ex.id for ex in examples[:5]],
    }


def _proofwriter_file(root: Path, split: str) -> Path:
    aliases = {"validation": "dev", "valid": "dev"}
    split = aliases.get(split, split)
    candidates = [root / f"data-{split}.jsonl", root / f"meta-{split}.jsonl"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _fairness_ready(report: dict[str, Any]) -> bool:
    datasets_ok = all(item.get("status") == "ok" and item.get("coverage", 0) > 0 for item in report["datasets"].values())
    methods = set(report["method_set"]["required_result_methods"])
    required = set(REQUIRED_LLM_METHODS)
    return datasets_ok and required.issubset(methods)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local no-API fairness preflight checks for CAGE/SymbCoT+CAGE experiments.")
    parser.add_argument("--proofwriter-root", default=DEFAULT_PROOFWRITER_ROOT)
    parser.add_argument("--prontoqa-path", default=DEFAULT_PRONTOQA_PATH)
    parser.add_argument("--folio-path", default=DEFAULT_FOLIO_PATH)
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/preflight")
    args = parser.parse_args()

    report = build_preflight_report(args.proofwriter_root, args.prontoqa_path, args.folio_path, args.n_examples)
    write_preflight_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote preflight outputs to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
