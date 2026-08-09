from __future__ import annotations

import json
import os
import re
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd

from .cage import CAGEConfig, cage_predict
from .cage_select import CAGESelectConfig, cage_select_predict
from .counterfactuals import generate_counterfactuals
from .datasets import FolioExample, load_folio_examples, load_prontoqa_examples, load_proofwriter_examples
from .evaluate import plot_metrics, summarize_base, summarize_counterfactual
from .folio_cage import FolioCAGEConfig, folio_cage_predict, folio_cage_select_predict, folio_cpa_predict
from .generator import generate_examples, write_jsonl
from .llm_baselines import LLMBaselineConfig, folio_llm_baseline_rows, llm_baseline_predictions, normalize_baseline_methods
from .prompts import cpa_prompt, direct_prompt
from .schema import CounterfactualExample, Example, Prediction

VALID_LABELS = {"true", "false", "unknown"}
ANSWER_KEYS = ("answer", "final_answer", "final_label", "label", "prediction", "verdict", "result", "correct_label", "corrected_answer", "truth_value", "validity")
NESTED_ANSWER_KEYS = ("revised_answer", "revised_draft", "revision", "corrected")
CONTAINER_KEYS = {"candidates", "primary_strategy", "check_negation", "check_contradiction", "support_minimization"}
RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def _llm_retry_count() -> int:
    return max(1, int(os.environ.get("CF_REASONING_LLM_RETRIES", "3")))


def _llm_retry_delay() -> float:
    return max(0.0, float(os.environ.get("CF_REASONING_LLM_RETRY_DELAY", "1")))


def _retry_delay(attempt: int) -> float:
    return _llm_retry_delay() * (2**attempt)


def _is_retryable_openai_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return int(status_code) in RETRYABLE_HTTP_STATUS
    name = exc.__class__.__name__.lower()
    return any(token in name for token in ("connection", "timeout", "ratelimit", "internalserver"))


def _call_with_retries(operation, is_retryable, error_prefix: str):
    last_error = None
    retry_limit = _llm_retry_count()
    for attempt in range(retry_limit):
        try:
            return operation()
        except Exception as exc:
            if not is_retryable(exc):
                raise
            last_error = exc
            if attempt + 1 == retry_limit:
                break
            time.sleep(_retry_delay(attempt))
    raise last_error
CPA_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["true", "false", "unknown"]},
        "causal_premises": {"type": "array", "items": {"type": "string"}},
        "brief_explanation": {"type": "string"},
    },
    "required": ["answer", "causal_premises", "brief_explanation"],
    "additionalProperties": False,
}
FOLIO_STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_strategy": {"type": "string", "enum": ["direct_fol_check", "proof_by_contradiction", "case_split", "unknown_search"]},
        "check_negation": {"type": "boolean"},
        "check_consistency": {"type": "boolean"},
        "reason": {"type": "string"},
        "answer": {"type": "string", "enum": ["true", "false", "unknown"]},
    },
    "required": ["primary_strategy", "check_negation", "check_consistency", "reason", "answer"],
    "additionalProperties": False,
}


def _json_from_text(text: str):
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _extract_label(text: str) -> str:
    data = _json_from_text(text)
    label = _label_from_data(data, mode="general")
    if label in VALID_LABELS:
        return label
    lowered = text.strip().lower()
    conclusion_matches = re.findall(r"\b(?:query|answer|conclusion|statement)\s+(?:is|=)\s+(true|false|unknown|yes|no|entailed|not\s+entailed|contradiction)\b", lowered)
    if conclusion_matches:
        return _normalize_general_label(conclusion_matches[-1])
    for label in VALID_LABELS:
        if lowered == label or lowered.endswith(label):
            return label
    return _normalize_general_label(lowered)


def _extract_folio_label(text: str) -> str:
    data = _json_from_text(text)
    label = _label_from_data(data, mode="folio")
    if label in VALID_LABELS:
        return label
    label = _normalize_folio_label(text.strip())
    if label in VALID_LABELS:
        return label
    matches = re.findall(
        r"\b(true|false|unknown|entailment|entailed|entails|valid|invalid|not_valid|not\s+valid|contradiction|contradicts|contradictory|neutral|unsupported|not_entailed|not\s+entailed|uncertain|inconclusive)\b",
        text.lower(),
    )
    return _normalize_folio_label(matches[-1]) if matches else "invalid"


def _label_from_data(data, mode: str) -> str:
    if isinstance(data, bool):
        return "true" if data else "false"
    if isinstance(data, str):
        return _normalize_label(data, mode)
    if not isinstance(data, dict):
        return "invalid"
    if any(key in data for key in CONTAINER_KEYS) and not any(key in data for key in ANSWER_KEYS + NESTED_ANSWER_KEYS + ("valid", "conclusion_entailed", "correct")):
        return "invalid"
    for key in NESTED_ANSWER_KEYS:
        if key in data:
            nested = _label_from_data(data[key], mode)
            if nested in VALID_LABELS:
                return nested
    for key in ANSWER_KEYS:
        if key in data:
            label = _normalize_label(str(data[key]), mode)
            if label in VALID_LABELS:
                return label
    if isinstance(data.get("valid"), bool):
        if data["valid"]:
            return "true"
        return "unknown" if mode == "folio" else "false"
    if isinstance(data.get("conclusion_entailed"), bool):
        if data["conclusion_entailed"]:
            return "true"
        return "unknown" if mode == "folio" else "false"
    if isinstance(data.get("correct"), bool):
        return "true" if data["correct"] else "false"
    return "invalid"


def _normalize_label(label: str, mode: str) -> str:
    if mode == "folio":
        return _normalize_folio_label(label)
    return _normalize_general_label(label)


def _normalize_general_label(label: str) -> str:
    normalized = _clean_label(label)
    aliases = {
        "yes": "true",
        "y": "true",
        "entailed": "true",
        "entailment": "true",
        "entails": "true",
        "proved": "true",
        "proven": "true",
        "follows": "true",
        "valid": "true",
        "no": "false",
        "n": "false",
        "contradiction": "false",
        "contradicts": "false",
        "contradictory": "false",
        "negation": "false",
        "not_true": "false",
        "not true": "false",
        "undetermined": "unknown",
        "uncertain": "unknown",
        "inconclusive": "unknown",
        "not_entailed": "unknown",
        "not entailed": "unknown",
        "unsupported": "unknown",
        "cannot_determine": "unknown",
        "cannot determine": "unknown",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in VALID_LABELS else "invalid"


def _normalize_folio_label(label: str) -> str:
    normalized = _clean_label(label)
    aliases = {
        "entailment": "true",
        "entailed": "true",
        "entails": "true",
        "valid": "true",
        "contradiction": "false",
        "contradicts": "false",
        "contradictory": "false",
        "neutral": "unknown",
        "unsupported": "unknown",
        "not_entailed": "unknown",
        "not entailed": "unknown",
        "not_valid": "unknown",
        "not valid": "unknown",
        "invalid": "unknown",
        "uncertain": "unknown",
        "inconclusive": "unknown",
        "undetermined": "unknown",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in VALID_LABELS else "invalid"


def _clean_label(label: str) -> str:
    cleaned = label.strip().lower().strip("`*_ .:;!\"'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _extract_premises(text: str) -> tuple[str, ...]:
    data = _json_from_text(text)
    if not isinstance(data, dict):
        return ()
    values = data.get("causal_premises", [])
    if isinstance(values, list):
        return tuple(str(v) for v in values)
    return ()


def _call_anthropic(prompt: str, model: str, max_tokens: int, json_output: bool = False, schema: dict | None = None) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Install optional dependency with: pip install -e .[llm]") from exc

    client = anthropic.Anthropic()
    kwargs = {}
    if json_output:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema or CPA_SCHEMA}}

    def request():
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    try:
        response = _call_with_retries(
            request,
            lambda exc: isinstance(exc, (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError))
            and (
                not isinstance(exc, anthropic.APIStatusError)
                or getattr(exc, "status_code", None) in RETRYABLE_HTTP_STATUS
            ),
            "Anthropic API error",
        )
    except anthropic.AuthenticationError:
        raise RuntimeError("Anthropic authentication failed. Check ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or active Anthropic profile.")
    except TypeError as exc:
        if "Could not resolve authentication method" in str(exc):
            raise RuntimeError("Anthropic authentication is not configured. Set ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or use an active Anthropic profile.") from exc
        raise
    except anthropic.APIConnectionError as exc:
        raise RuntimeError("Anthropic API connection failed. Check your network and retry.") from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    if response.stop_reason == "refusal":
        return "invalid"
    return "\n".join(block.text for block in response.content if block.type == "text")



def _call_openai_compatible(
    prompt: str,
    model: str,
    max_tokens: int,
    api_key: str,
    base_url: str | None = None,
    json_output: bool = False,
    schema: dict | None = None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install OpenAI-compatible dependency with: pip install openai") from exc
    client = OpenAI(api_key=api_key, base_url=base_url)
    content = prompt
    kwargs = {}
    if json_output:
        content += "\n\nReturn a single valid JSON object only. Do not wrap it in Markdown."
        kwargs["response_format"] = {"type": "json_object"}

    def request():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=float(os.environ.get("CF_REASONING_TEMPERATURE", "0")),
            **kwargs,
        )

    try:
        response = _call_with_retries(
            request,
            _is_retryable_openai_error,
            "OpenAI-compatible API error",
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI-compatible API error: {exc}") from exc
    message = response.choices[0].message.content if response.choices else ""
    return message or ""


def _call_deepseek(prompt: str, model: str, max_tokens: int, json_output: bool = False, schema: dict | None = None) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
    if not api_key:
        raise RuntimeError("DeepSeek authentication is not configured. Set DEEPSEEK_API_KEY.")
    return _call_openai_compatible(
        prompt,
        model=model,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        json_output=json_output,
        schema=schema,
    )


def _canonical_method(method: str) -> str:
    aliases = {
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
    return aliases.get(method, method)


def _canonicalize_results(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["benchmark"] = benchmark
    out["method"] = out["method"].map(_canonical_method)
    cols = ["benchmark", "split", "method", "example_id", "gold", "pred", "accuracy"]
    return out[[col for col in cols if col in out.columns] + [col for col in out.columns if col not in cols]]


def _write_unified_llm_outputs(df: pd.DataFrame, output_dir: Path, benchmark: str) -> None:
    unified = _canonicalize_results(df, benchmark)
    unified.to_csv(output_dir / "results_llm.csv", index=False)
    if unified.empty:
        pd.DataFrame().to_csv(output_dir / "summary_llm.csv", index=False)
        return
    unified.groupby(["benchmark", "split", "method"], dropna=False)[["accuracy"]].mean().reset_index().to_csv(output_dir / "summary_llm.csv", index=False)


def _call_llm(prompt: str, max_tokens: int, json_output: bool = False, schema: dict | None = None) -> str:
    provider = os.environ.get("CF_REASONING_LLM_PROVIDER", "anthropic").strip().lower()
    if provider == "deepseek":
        model = os.environ.get("DEEPSEEK_MODEL", os.environ.get("CF_REASONING_MODEL", "deepseek-chat"))
        return _call_deepseek(prompt, model=model, max_tokens=max_tokens, json_output=json_output, schema=schema)
    if provider in {"openai-compatible", "openai_compatible", "openai"}:
        api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI-compatible authentication is not configured. Set OPENAI_COMPATIBLE_API_KEY or OPENAI_API_KEY.")
        model = os.environ.get("OPENAI_COMPATIBLE_MODEL", os.environ.get("CF_REASONING_MODEL", "gpt-4o-mini"))
        return _call_openai_compatible(
            prompt,
            model=model,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=os.environ.get("OPENAI_COMPATIBLE_BASE_URL"),
            json_output=json_output,
            schema=schema,
        )
    if provider == "anthropic":
        model = os.environ.get("CF_REASONING_MODEL", "claude-opus-4-8")
        return _call_anthropic(prompt, model=model, max_tokens=max_tokens, json_output=json_output, schema=schema)
    raise RuntimeError(f"Unsupported LLM provider: {provider}. Use anthropic, deepseek, or openai-compatible.")


def _llm_predictions(
    examples: list[Example] | list[CounterfactualExample],
    max_tokens: int,
    baseline_methods: tuple[str, ...] = (),
) -> tuple[list[Prediction], list[dict[str, str]]]:
    preds: list[Prediction] = []
    raw_rows: list[dict[str, str]] = []
    base_cache = {}
    for ex in examples:
        for method, build_prompt, json_output in [
            ("llm_direct", direct_prompt, False),
            ("llm_cpa", cpa_prompt, True),
        ]:
            raw = _call_llm(build_prompt(ex), max_tokens=max_tokens, json_output=json_output)
            pred = Prediction(ex.id, method, _extract_label(raw), _extract_premises(raw), raw)
            preds.append(pred)
            raw_row = {"example_id": ex.id, "method": method, "raw_response": raw}
            raw_rows.append(raw_row)
            base_cache[(ex.id, method)] = (pred, [raw_row])

        cage_pred, cage_raw = cage_predict(
            ex,
            lambda prompt, tokens, json_output, schema: _call_llm(prompt, max_tokens=tokens, json_output=json_output, schema=schema),
            _extract_label,
            _extract_premises,
            CAGEConfig(max_counterfactuals=3, max_tokens=max_tokens),
        )
        preds.append(cage_pred)
        raw_rows.extend(cage_raw)

        cage_select_pred, cage_select_raw = cage_select_predict(
            ex,
            lambda prompt, tokens, json_output, schema: _call_llm(prompt, max_tokens=tokens, json_output=json_output, schema=schema),
            _extract_label,
            _extract_premises,
            CAGESelectConfig(n_candidates=3, max_counterfactuals=3, max_tokens=max_tokens),
        )
        preds.append(cage_select_pred)
        raw_rows.extend(cage_select_raw)
    if baseline_methods:
        baseline_preds, baseline_raw = llm_baseline_predictions(
            examples,
            baseline_methods,
            lambda prompt, tokens, json_output, schema: _call_llm(prompt, max_tokens=tokens, json_output=json_output, schema=schema),
            _extract_label,
            _extract_premises,
            LLMBaselineConfig(max_tokens=max_tokens),
            base_predictions=base_cache,
        )
        preds.extend(baseline_preds)
        raw_rows.extend(baseline_raw)
    return preds, raw_rows


def _llm_baseline_only_predictions(
    examples: list[Example] | list[CounterfactualExample],
    max_tokens: int,
    baseline_methods: tuple[str, ...],
) -> tuple[list[Prediction], list[dict[str, str]]]:
    base_cache = {}
    needs_direct = any(method.startswith("direct_cage") for method in baseline_methods)
    if needs_direct:
        for ex in examples:
            raw = _call_llm(direct_prompt(ex), max_tokens=max_tokens, json_output=False)
            pred = Prediction(ex.id, "llm_direct", _extract_label(raw), _extract_premises(raw), raw)
            raw_row = {"example_id": ex.id, "method": "llm_direct", "raw_response": raw}
            base_cache[(ex.id, "llm_direct")] = (pred, [raw_row])
    return llm_baseline_predictions(
        examples,
        baseline_methods,
        lambda prompt, tokens, json_output, schema: _call_llm(prompt, max_tokens=tokens, json_output=json_output, schema=schema),
        _extract_label,
        _extract_premises,
        LLMBaselineConfig(max_tokens=max_tokens),
        base_predictions=base_cache,
    )


def _proofwriter_file(root: str | Path, split: str) -> Path:
    root = Path(root)
    for candidate in [root / f"data-{split}.jsonl", root / f"meta-{split}.jsonl"]:
        if candidate.exists():
            return candidate
    return root / f"data-{split}.jsonl"


def _load_llm_examples(args: Namespace, n: int) -> list[Example]:
    if getattr(args, "dataset", "") == "prontoqa":
        examples, report = load_prontoqa_examples(getattr(args, "prontoqa_path", "data/raw/prontoqa2/ProntoQA_dev_gpt-4.json"), limit=n, split="prontoqa")
        if examples:
            print(
                f"Loaded {len(examples)} PrOntoQA examples for optional LLM experiment from {Path(getattr(args, 'prontoqa_path', '')).name} "
                f"({report.skipped} skipped, coverage={report.parsed / report.loaded if report.loaded else 0:.3f})."
            )
            return examples
        print(f"No parseable PrOntoQA examples for optional LLM experiment ({report.skipped} skipped); falling back to synthetic data.")
    proofwriter_root = getattr(args, "proofwriter_root", None)
    proofwriter_path = getattr(args, "proofwriter_path", None)
    if proofwriter_root:
        path = _proofwriter_file(proofwriter_root, "test")
        examples, report = load_proofwriter_examples(path, limit=n, split="test")
        if examples:
            print(f"Loaded {len(examples)} ProofWriter examples for optional LLM experiment from {Path(path).name}.")
            return examples
        print(f"No parseable ProofWriter examples for optional LLM experiment ({report.skipped} skipped); falling back to synthetic data.")
    if proofwriter_path:
        examples, report = load_proofwriter_examples(proofwriter_path, limit=n, split="proofwriter")
        if examples:
            print(f"Loaded {len(examples)} ProofWriter examples for optional LLM experiment from {Path(proofwriter_path).name}.")
            return examples
        print(f"No parseable ProofWriter examples for optional LLM experiment ({report.skipped} skipped); falling back to synthetic data.")
    examples = generate_examples(n, seed=getattr(args, "seed", 42), max_depth=getattr(args, "max_depth", 3))
    for ex in examples:
        ex.split = "synthetic"
    return examples


def _folio_direct_prompt(example: FolioExample, use_fol: bool = False) -> str:
    fol = ""
    if use_fol:
        fol_lines = [f"FOL{i + 1}: {premise}" for i, premise in enumerate(example.premises_fol)]
        fol_lines.append(f"Conclusion-FOL: {example.conclusion_fol}")
        fol = "\n\nFormal representation:\n" + "\n".join(fol_lines)
    return f"""You are solving a FOLIO logical reasoning problem.

Premises:
{example.text}
{fol}

Answer with exactly one of: true, false, unknown.
Final answer:"""


def _folio_strategy_prompt(example: FolioExample) -> str:
    fol_lines = [f"FOL{i + 1}: {premise}" for i, premise in enumerate(example.premises_fol)]
    fol_lines.append(f"Conclusion-FOL: {example.conclusion_fol}")
    return f"""You are a meta-cognitive planner for a FOL logical reasoning problem.

Choose the reasoning strategy and provide the final label. Use the formal representation when helpful, but do not output hidden chain-of-thought.

Natural-language problem:
{example.text}

Formal representation:
{chr(10).join(fol_lines)}

Return only JSON matching the requested schema."""


def _run_folio_llm(args: Namespace, max_tokens: int, output_dir: Path, baseline_methods: tuple[str, ...] = ()) -> bool:
    if getattr(args, "dataset", "") != "folio":
        return False
    n = min(getattr(args, "n_examples", 204), 204)
    examples = load_folio_examples(getattr(args, "folio_path", "data/raw/FOLIO/data/v0.0/folio-validation.jsonl"), limit=n, split="folio")
    if not examples:
        print("Skipping FOLIO LLM experiment: no examples found.")
        return True
    rows = []
    raw_rows = []
    base_cache = {}
    try:
        for ex in examples:
            for method, prompt, json_output, schema in [
                ("llm_direct", _folio_direct_prompt(ex, use_fol=False), False, None),
                ("folio_llm_fol", _folio_direct_prompt(ex, use_fol=True), False, None),
                ("folio_llm_strategy", _folio_strategy_prompt(ex), True, FOLIO_STRATEGY_SCHEMA),
            ]:
                raw = _call_llm(prompt, max_tokens=max_tokens, json_output=json_output, schema=schema)
                pred = _extract_folio_label(raw)
                rows.append({"split": ex.split, "example_id": ex.id, "method": method, "gold": ex.label, "pred": pred, "accuracy": int(pred == ex.label)})
                raw_row = {"example_id": ex.id, "method": method, "raw_response": raw}
                raw_rows.append(raw_row)
                if method == "llm_direct":
                    base_cache[(ex.id, "llm_direct")] = (Prediction(ex.id, "llm_direct", pred, (), raw), [raw_row])

            for pred, pred_raw in [
                folio_cpa_predict(
                    ex,
                    lambda prompt, tokens, json_output, schema: _call_llm(prompt, max_tokens=tokens, json_output=json_output, schema=schema),
                    _extract_folio_label,
                    FolioCAGEConfig(max_tokens=max_tokens),
                ),
                folio_cage_predict(
                    ex,
                    lambda prompt, tokens, json_output, schema: _call_llm(prompt, max_tokens=tokens, json_output=json_output, schema=schema),
                    _extract_folio_label,
                    FolioCAGEConfig(max_tokens=max_tokens),
                ),
                folio_cage_select_predict(
                    ex,
                    lambda prompt, tokens, json_output, schema: _call_llm(prompt, max_tokens=tokens, json_output=json_output, schema=schema),
                    _extract_folio_label,
                    FolioCAGEConfig(max_tokens=max_tokens),
                ),
            ]:
                rows.append({"split": ex.split, "example_id": ex.id, "method": pred.method, "gold": ex.label, "pred": pred.label, "accuracy": int(pred.label == ex.label)})
                raw_rows.extend(pred_raw)
        if baseline_methods:
            baseline_rows, baseline_raw = folio_llm_baseline_rows(
                examples,
                baseline_methods,
                lambda prompt, tokens, json_output, schema: _call_llm(prompt, max_tokens=tokens, json_output=json_output, schema=schema),
                _extract_folio_label,
                LLMBaselineConfig(max_tokens=max_tokens),
                base_predictions=base_cache,
            )
            rows.extend(baseline_rows)
            raw_rows.extend(baseline_raw)
    except RuntimeError as exc:
        print(f"Skipping FOLIO LLM experiment: {exc}")
        return True
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results_folio_llm.csv", index=False)
    df.groupby(["split", "method"])[["accuracy"]].mean().reset_index().to_csv(output_dir / "summary_folio_llm.csv", index=False)
    _write_unified_llm_outputs(df, output_dir, "folio")
    write_jsonl(output_dir / "folio_llm_raw_outputs.jsonl", raw_rows)
    print(f"Wrote FOLIO LLM outputs to {_ascii_path(output_dir)}.")
    return True


def _ascii_path(path: Path) -> str:
    return path.as_posix().encode("unicode_escape").decode("ascii")


def run_llm_if_available(args: Namespace) -> None:
    n = getattr(args, "n_examples", 20)
    max_tokens = int(os.environ.get("CF_REASONING_MAX_TOKENS", "512"))
    output_dir = Path(getattr(args, "output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        baseline_methods = normalize_baseline_methods(getattr(args, "llm_baseline_methods", ""))
    except ValueError as exc:
        print(f"Skipping LLM experiment: {exc}")
        return
    if _run_folio_llm(args, max_tokens, output_dir, baseline_methods):
        return
    examples = _load_llm_examples(args, n)
    cfs = generate_counterfactuals(examples, seed=getattr(args, "seed", 42), max_per_example=2)

    try:
        if baseline_methods:
            base_preds, base_raw = _llm_baseline_only_predictions(examples, max_tokens, baseline_methods)
            cf_preds, cf_raw = _llm_baseline_only_predictions(cfs, max_tokens, baseline_methods)
        else:
            base_preds, base_raw = _llm_predictions(examples, max_tokens)
            cf_preds, cf_raw = _llm_predictions(cfs, max_tokens)
    except RuntimeError as exc:
        print(f"Skipping LLM experiment: {exc}")
        return

    base_df = summarize_base(examples, base_preds)
    cf_df = summarize_counterfactual(examples, cfs, cf_preds)
    base_df.to_csv(output_dir / "results_base_llm.csv", index=False)
    cf_df.to_csv(output_dir / "results_counterfactual_llm.csv", index=False)
    benchmark = getattr(args, "dataset", "") or examples[0].split if examples else "unknown"
    benchmark = "synthetic" if benchmark in {"auto", ""} else benchmark
    _write_unified_llm_outputs(base_df, output_dir, benchmark)
    plot_metrics(base_df, cf_df, output_dir / "figures_llm")
    write_jsonl(output_dir / "llm_raw_outputs.jsonl", base_raw + cf_raw)
    write_jsonl(output_dir / "llm_predictions.jsonl", base_preds + cf_preds)
    print(f"Wrote optional LLM outputs to {_ascii_path(output_dir)}.")
