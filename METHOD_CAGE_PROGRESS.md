# Method-agnostic CAGE progress

## Current objective

Continue the method-agnostic CAGE wrapper work after the three 20-example smoke runs completed.

## Starting findings from previous conversation

- Three smoke output directories exist and have complete result coverage:
  - `outputs/smoke_proofwriter_20_method_cage/`
  - `outputs/smoke_prontoqa2_20_method_cage/`
  - `outputs/smoke_folio_20_method_cage/`
- Core comparison methods are present:
  - `llm_direct` vs `llm_direct_cage`
  - `llm_logiclm` vs `llm_logiclm_cage`
  - `llm_symbcot` vs `llm_symbcot_cage`
- Invalid rates improved to zero for CAGE wrappers.
- Main blocker: the conservative label policy is not fully enforced. Valid base labels were changed by several wrappers:
  - ProofWriter: Direct+CAGE 0, LogicLM+CAGE 2, SymbCoT+CAGE 9
  - PrOntoQA: Direct+CAGE 0, LogicLM+CAGE 2, SymbCoT+CAGE 4
  - FOLIO: Direct+CAGE 1, LogicLM+CAGE 2, SymbCoT+CAGE 8
- Recommendation before scaling to 100 examples: fix wrappers so valid initial labels are preserved and only invalid initial labels can trigger answer repair.

## Implementation constraints

- Keep provider-agnostic design.
- Do not add Anthropic/Claude SDK code or provider-specific calls.
- Use the existing injected `call_llm(prompt, max_tokens, json_output, schema)` path.
- Record diagnostics in trace/raw response while preserving labels for valid base predictions.

## Work log

### 2026-08-07

- Created this progress README to preserve state across context switches.
- Next: inspect `src/cf_reasoning/method_cage.py`, `src/cf_reasoning/llm_baselines.py`, `src/cf_reasoning/folio_cage.py`, and tests to locate non-conservative paths.
- Inspection finding: `src/cf_reasoning/method_cage.py` already preserves valid labels; the non-conservative smoke comparison came from dispatch/caching issues and FOLIO SymbCoT using the older transfer-repair implementation.
- Updated `src/cf_reasoning/llm_baselines.py` so `logiclm`, `symbcot`, `direct_cage`, `logiclm_cage`, and `symbcot_cage` share a `BaseCache` per example.
- Updated general `symbcot_cage` dispatch to use `structured_cage_wrap`, and FOLIO `symbcot_cage` dispatch to use `folio_cage_wrap`; valid initial labels now only receive diagnostics, not label repair.
- Updated `src/cf_reasoning/llm_client.py` to pass existing default `llm_direct` predictions into baseline wrappers via `base_predictions`, avoiding separate stochastic Direct calls for `llm_direct_cage`.
- Cleaned duplicate `_llm_predictions` definition in `src/cf_reasoning/llm_client.py` and added FOLIO `llm_direct` cache population for FOLIO wrappers.
- Next: add tests that prove wrapper/base pair preservation when base methods and caged methods are requested together, including FOLIO SymbCoT+CAGE.
- Added tests in `tests/test_llm_baselines.py` for:
  - LogicLM+CAGE reusing the already requested LogicLM base prediction.
  - SymbCoT+CAGE preserving a valid SymbCoT base label even when probes disagree.
  - FOLIO Direct/LogicLM/SymbCoT CAGE wrappers preserving valid labels with transfer diagnostics.
- Ran `python -m pytest tests/test_llm_baselines.py -q`: 14 passed.
- Next: run related adapter/fair-eval tests, then the full suite if those pass.
- Ran `python -m pytest tests/test_llm_baselines.py tests/test_symbcot_adapter.py tests/test_fair_eval.py -q`: 22 passed.
- Ran `python -m pytest -q`: 81 passed.
- Checked local data paths:
  - ProofWriter: `data/raw/proofwriter/data-test.jsonl`
  - PrOntoQA: `data/raw/prontoqa2/ProntoQA_dev_gpt-4.json`
  - FOLIO: `data/raw/FOLIO/data/v0.0/folio-validation.jsonl`
- Checked LLM provider env presence without printing secrets: no `CF_REASONING_LLM_PROVIDER`, Anthropic, DeepSeek, or OpenAI-compatible credentials are set in the current shell. Smoke reruns need provider env configured first.
- Next when credentials are available: rerun the three 20-example smoke commands into fresh output directories and verify pairwise preservation again.

## Verification so far

- Unit/targeted tests passed:
  - `python -m pytest tests/test_llm_baselines.py -q` -> 14 passed
  - `python -m pytest tests/test_llm_baselines.py tests/test_symbcot_adapter.py tests/test_fair_eval.py -q` -> 22 passed
  - `python -m pytest -q` -> 81 passed
- Local no-API cache-preservation smoke passed for general Direct+CAGE and FOLIO Direct+CAGE using injected `base_predictions`.
- Three API smoke reruns are blocked in the current shell because no provider credentials/env vars are set.

## Commands to rerun when LLM env is configured

```bash
python -m cf_reasoning.run_experiment --llm --dataset proofwriter --proofwriter-path data/raw/proofwriter/data-test.jsonl --n-examples 20 --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage --output-dir outputs/smoke_proofwriter_20_method_cage_conservative_fix
```

```bash
python -m cf_reasoning.run_experiment --llm --dataset prontoqa --prontoqa-path data/raw/prontoqa2/ProntoQA_dev_gpt-4.json --n-examples 20 --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage --output-dir outputs/smoke_prontoqa2_20_method_cage_conservative_fix
```

```bash
python -m cf_reasoning.run_experiment --llm --dataset folio --folio-path data/raw/FOLIO/data/v0.0/folio-validation.jsonl --n-examples 20 --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage --output-dir outputs/smoke_folio_20_method_cage_conservative_fix
```

## Expected post-rerun checks

- Method coverage includes `llm_direct`, `llm_direct_cage`, `llm_logiclm`, `llm_logiclm_cage`, `llm_symbcot`, `llm_symbcot_cage`.
- For every example where the base method has a valid label, the corresponding CAGE wrapper must have the same label.
- `repair_triggered` should be true only when `initial_answer` is invalid.
- FOLIO wrapper traces should include transfer diagnostics while preserving valid labels.

## Conservative-fix 20-example smoke results

Rerun output directories:

- `outputs/smoke_proofwriter_20_method_cage_conservative_fix/`
- `outputs/smoke_prontoqa2_20_method_cage_conservative_fix/`
- `outputs/smoke_folio_20_method_cage_conservative_fix/`

All three contain complete method coverage for the six pairwise methods:

- `llm_direct`, `llm_direct_cage`
- `llm_logiclm`, `llm_logiclm_cage`
- `llm_symbcot`, `llm_symbcot_cage`

Accuracy summary:

| Dataset | Direct | Direct+CAGE | LogicLM | LogicLM+CAGE | SymbCoT | SymbCoT+CAGE |
|---|---:|---:|---:|---:|---:|---:|
| ProofWriter | 0.55 | 0.55 | 0.65 | 0.65 | 0.35 | 0.35 |
| PrOntoQA | 0.35 | 0.40 | 0.60 | 0.60 | 0.25 | 0.25 |
| FOLIO | 0.55 | 0.55 | 0.75 | 0.75 | 0.45 | 0.45 |

Invalid rate summary:

- ProofWriter: all six methods 0.00.
- PrOntoQA: `llm_direct` 0.05 -> `llm_direct_cage` 0.00; all other methods 0.00.
- FOLIO: all six methods 0.00.

Conservative policy check:

- ProofWriter: valid-label changed counts are 0 for Direct+CAGE, LogicLM+CAGE, and SymbCoT+CAGE.
- PrOntoQA: valid-label changed counts are 0 for Direct+CAGE, LogicLM+CAGE, and SymbCoT+CAGE.
- FOLIO: valid-label changed counts are 0 for Direct+CAGE, LogicLM+CAGE, and SymbCoT+CAGE.

Trace stats from `llm_predictions.jsonl` for ProofWriter/PrOntoQA:

| Dataset | Method | repair_rate | valid_preservation | avg diag ok rate |
|---|---|---:|---:|---:|
| ProofWriter | llm_direct_cage | 0.033 | 1.0 | 0.533 |
| ProofWriter | llm_logiclm_cage | 0.000 | 1.0 | 0.617 |
| ProofWriter | llm_symbcot_cage | 0.000 | 1.0 | 0.683 |
| PrOntoQA | llm_direct_cage | 0.017 | 1.0 | 0.567 |
| PrOntoQA | llm_logiclm_cage | 0.000 | 1.0 | 0.633 |
| PrOntoQA | llm_symbcot_cage | 0.000 | 1.0 | 0.550 |

FOLIO has no wrapper repair raw rows, consistent with no invalid initial labels and conservative no-repair behavior.

Conclusion: the conservative wrapper fix worked. These 20-example smokes now satisfy the planned label-preservation condition. The result now measures diagnostics + invalid-label repair, not aggressive answer repair. Scaling to 100 is reasonable if the next experiment goal is conservative method-agnostic CAGE; expect accuracy to mostly match the base methods except where invalid outputs are repaired.

## Gated smoke commands

Use these commands for the next 20-example comparison run once LLM credentials are configured:

```bash
python -m cf_reasoning.run_experiment --llm --dataset proofwriter --proofwriter-path data/raw/proofwriter/data-test.jsonl --n-examples 20 --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage,direct_cage_gated,logiclm_cage_gated,symbcot_cage_gated --output-dir outputs/smoke_proofwriter_20_method_cage_gated
```

```bash
python -m cf_reasoning.run_experiment --llm --dataset prontoqa --prontoqa-path data/raw/prontoqa2/ProntoQA_dev_gpt-4.json --n-examples 20 --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage,direct_cage_gated,logiclm_cage_gated,symbcot_cage_gated --output-dir outputs/smoke_prontoqa2_20_method_cage_gated
```

```bash
python -m cf_reasoning.run_experiment --llm --dataset folio --folio-path data/raw/FOLIO/data/v0.0/folio-validation.jsonl --n-examples 20 --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage,direct_cage_gated,logiclm_cage_gated,symbcot_cage_gated --output-dir outputs/smoke_folio_20_method_cage_gated
```


- Added `structured_cage_wrap_gated` and `folio_cage_wrap_gated` in `src/cf_reasoning/method_cage.py`.
- Conservative wrappers are unchanged: `direct_cage`, `logiclm_cage`, and `symbcot_cage` still only repair invalid initial labels.
- Gated wrappers can repair valid initial labels only when diagnostics have a strict alternative-label majority:
  - initial label must be valid, otherwise repair follows the existing invalid-label path;
  - at least two failed diagnostics are required;
  - failed diagnostics must contain a strict majority valid alternative label different from the initial label;
  - the repair output is accepted only if its label exactly matches that gated target label.
- Added baseline dispatch in `src/cf_reasoning/llm_baselines.py` for general datasets and FOLIO.
- Updated `normalize_baseline_methods("all")` coverage to include gated methods.

Verification:

- `python -m compileall src/cf_reasoning/method_cage.py src/cf_reasoning/llm_baselines.py` -> passed.
- `python -m pytest tests/test_llm_baselines.py -q` -> 15 passed.
- `python -m pytest tests/test_llm_baselines.py tests/test_symbcot_adapter.py tests/test_fair_eval.py -q` -> 23 passed.
- `python -m pytest -q` -> 82 passed.

Next recommended experiment:

Run 20-example smoke tests including both conservative and gated variants, then compare whether gated improves accuracy without reintroducing large valid-label damage. Suggested method list:

```bash
--llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage,direct_cage_gated,logiclm_cage_gated,symbcot_cage_gated
```

## Next-step execution checklist

1. Confirm the LLM provider environment is configured before spending API tokens. Do not start a smoke run if the provider or credentials are missing.
2. Run the three 20-example commands above, writing each dataset to its own fresh `*_gated` output directory.
3. After each dataset finishes, immediately update this file with:
   - output directory and completion status;
   - method coverage and row counts;
   - accuracy for all nine methods;
   - invalid rate for all nine methods;
   - conservative valid-base-label changed counts;
   - gated valid-base-label changed and accepted-repair counts;
   - repair trigger/acceptance rates;
   - diagnostic count and diagnostic failure/majority statistics.
4. Inspect every gated row where `repair_accepted == true` and record whether the accepted label matches the gold label. Also inspect every valid base row whose gated label changed.
5. Compare gated against both the base and conservative wrapper for each dataset. The gated variant is promising only if it improves accuracy or reduces invalid outputs without a large increase in incorrect changes to valid base labels.
6. If the 20-example results are favorable, repeat the same nine-method comparison at 100 examples using fresh `*_100_method_cage_gated` output directories. If gated is neutral or harmful, keep it as an ablation rather than scaling it.
7. Once the smoke analysis is complete, add a short decision paragraph here before changing the paper experiment plan.

## Gated smoke result record

Status: all three 20-example gated smokes completed: ProofWriter, PrOntoQA, and FOLIO.

Expected method coverage per dataset:

| Base | Conservative wrapper | Gated wrapper |
|---|---|---|
| `llm_direct` | `llm_direct_cage` | `llm_direct_cage_gated` |
| `llm_logiclm` | `llm_logiclm_cage` | `llm_logiclm_cage_gated` |
| `llm_symbcot` | `llm_symbcot_cage` | `llm_symbcot_cage_gated` |

Fill this table after the three runs:

| Dataset | Method | Rows | Accuracy | Invalid rate | Repair trigger rate | Repair accepted rate | Valid-base label changes |
|---|---|---:|---:|---:|---:|---:|---:|
| ProofWriter | `llm_direct` | 20 | 0.45 | 0.10 | n/a | n/a | n/a |
| ProofWriter | `llm_direct_cage` | 20 | 0.50 | 0.00 | 0.10 | n/a | 0 |
| ProofWriter | `llm_direct_cage_gated` | 20 | 0.45 | 0.00 | 0.30 | 0.25 | 3 |
| ProofWriter | `llm_logiclm` | 20 | 0.55 | 0.00 | n/a | n/a | n/a |
| ProofWriter | `llm_logiclm_cage` | 20 | 0.55 | 0.00 | 0.00 | n/a | 0 |
| ProofWriter | `llm_logiclm_cage_gated` | 20 | 0.55 | 0.00 | 0.05 | 0.00 | 0 |
| ProofWriter | `llm_symbcot` | 20 | 0.40 | 0.00 | n/a | n/a | n/a |
| ProofWriter | `llm_symbcot_cage` | 20 | 0.40 | 0.00 | 0.00 | n/a | 0 |
| ProofWriter | `llm_symbcot_cage_gated` | 20 | 0.45 | 0.00 | 0.20 | 0.05 | 1 |
| PrOntoQA | `llm_direct` | 20 | 0.30 | 0.10 | n/a | n/a | n/a |
| PrOntoQA | `llm_direct_cage` | 20 | 0.40 | 0.00 | 0.10 | n/a | 0 |
| PrOntoQA | `llm_direct_cage_gated` | 20 | 0.40 | 0.00 | 0.10 | 0.10 | 0 |
| PrOntoQA | `llm_logiclm` | 20 | 0.65 | 0.00 | n/a | n/a | n/a |
| PrOntoQA | `llm_logiclm_cage` | 20 | 0.65 | 0.00 | 0.00 | n/a | 0 |
| PrOntoQA | `llm_logiclm_cage_gated` | 20 | 0.65 | 0.00 | 0.00 | 0.00 | 0 |
| PrOntoQA | `llm_symbcot` | 20 | 0.50 | 0.00 | n/a | n/a | n/a |
| PrOntoQA | `llm_symbcot_cage` | 20 | 0.50 | 0.00 | 0.00 | n/a | 0 |
| PrOntoQA | `llm_symbcot_cage_gated` | 20 | 0.50 | 0.00 | 0.00 | 0.00 | 0 |
| FOLIO | `llm_direct` | 20 | 0.55 | 0.00 | n/a | n/a | n/a |
| FOLIO | `llm_direct_cage` | 20 | 0.55 | 0.00 | 0.00 | 0.00 | 0 |
| FOLIO | `llm_direct_cage_gated` | 20 | 0.75 | 0.00 | 0.60 | 0.40 | 8 |
| FOLIO | `llm_logiclm` | 20 | 0.70 | 0.00 | n/a | n/a | n/a |
| FOLIO | `llm_logiclm_cage` | 20 | 0.70 | 0.00 | 0.00 | 0.00 | 0 |
| FOLIO | `llm_logiclm_cage_gated` | 20 | 0.70 | 0.00 | 0.10 | 0.10 | 2 |
| FOLIO | `llm_symbcot` | 20 | 0.65 | 0.00 | n/a | n/a | n/a |
| FOLIO | `llm_symbcot_cage` | 20 | 0.65 | 0.00 | 0.00 | 0.00 | 0 |
| FOLIO | `llm_symbcot_cage_gated` | 20 | 0.70 | 0.00 | 0.15 | 0.15 | 3 |

Decision rule for the 20-example smoke:

- **Scale to 100:** gated accuracy improves over the base/conservative comparison on at least one dataset, with no substantial increase in incorrect valid-label changes.
- **Keep as ablation:** gated changes some valid labels but produces no reliable accuracy gain, or its gains are too unstable for the main method.
- **Revisit the gate:** gated repair is frequently triggered but rarely accepted, diagnostics lack a strict majority, or accepted repairs are often wrong.

## Full-dataset conservative + gated experiment plan

Requested after all three 20-example gated smokes completed.

Dataset sizes checked locally:

| Dataset | Path | Full size |
|---|---|---:|
| ProofWriter | `data/raw/proofwriter/data-test.jsonl` | 11820 |
| PrOntoQA | `data/raw/prontoqa2/ProntoQA_dev_gpt-4.json` | 500 |
| FOLIO | `data/raw/FOLIO/data/v0.0/folio-validation.jsonl` | 204 |

Important cost/runtime note:

- FOLIO and PrOntoQA are reasonable full runs.
- ProofWriter full test has 11820 examples. With base + conservative + gated wrappers, this can require a very large number of LLM calls because each CAGE wrapper runs diagnostics and some repairs. Before running ProofWriter full, consider whether to run a 100/500 subset first or confirm API budget explicitly.

Full-run method list:

```bash
--llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage,direct_cage_gated,logiclm_cage_gated,symbcot_cage_gated
```

Recommended execution order:

1. FOLIO full, because it is only 204 examples and gave the clearest gated signal.
2. PrOntoQA full, because it is 500 examples and tests whether gated remains mostly neutral/safe.
3. ProofWriter full only after confirming budget/runtime, because it is 11820 examples.

Commands:

```bash
python -m cf_reasoning.run_experiment --llm --dataset folio --folio-path data/raw/FOLIO/data/v0.0/folio-validation.jsonl --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage,direct_cage_gated,logiclm_cage_gated,symbcot_cage_gated --output-dir outputs/full_folio_method_cage_conservative_gated
```

```bash
python -m cf_reasoning.run_experiment --llm --dataset prontoqa --prontoqa-path data/raw/prontoqa2/ProntoQA_dev_gpt-4.json --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage,direct_cage_gated,logiclm_cage_gated,symbcot_cage_gated --output-dir outputs/full_prontoqa2_method_cage_conservative_gated
```

```bash
python -m cf_reasoning.run_experiment --llm --dataset proofwriter --proofwriter-path data/raw/proofwriter/data-test.jsonl --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage,direct_cage_gated,logiclm_cage_gated,symbcot_cage_gated --output-dir outputs/full_proofwriter_method_cage_conservative_gated
```

Progress update rule:

- Update this file immediately after each dataset finishes.
- Record output directory, completion status, method coverage, accuracy, invalid rate, valid-label changes, repair trigger rate, accepted repair rate, and accepted-repair correctness.
- If a run fails or is interrupted, record the failure command, partial output directory, last available row counts, and whether it is safe to resume or should use a fresh directory.

Full-run result table:

| Dataset | Status | Output dir | Notes |
|---|---|---|---|
| FOLIO | completed | `outputs/full_folio_method_cage_conservative_gated/` | 204/204 full run completed |
| PrOntoQA | completed-partial | `outputs/full_prontoqa2_method_cage_conservative_gated/` | Output has 100 rows per method, not the expected 500 full examples |
| ProofWriter | completed-partial | `outputs/full_proofwriter_method_cage_conservative_gated/` | Output has 100 rows per method, not the expected 11820 full examples |

## Full-run audit results

Important coverage caveat:

- FOLIO is a true full-dataset run: 204 rows per expected method.
- PrOntoQA output has 100 rows per expected method, not 500. Treat this as a 100-example run, not a full run.
- ProofWriter output has 100 rows per expected method, not 11820. Treat this as a 100-example run, not a full run.

Accuracy summary:

| Dataset | Rows/method | Direct | Direct+CAGE | Direct+Gated | LogicLM | LogicLM+CAGE | LogicLM+Gated | SymbCoT | SymbCoT+CAGE | SymbCoT+Gated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ProofWriter | 100 | 0.47 | 0.51 | 0.48 | 0.72 | 0.72 | 0.69 | 0.51 | 0.51 | 0.51 |
| PrOntoQA | 100 | 0.43 | 0.43 | 0.44 | 0.68 | 0.68 | 0.65 | 0.43 | 0.43 | 0.51 |
| FOLIO | 204 | 0.441 | 0.441 | 0.672 | 0.637 | 0.637 | 0.676 | 0.471 | 0.480 | 0.642 |

Invalid-rate summary:

| Dataset | Direct | Direct+CAGE | Direct+Gated | LogicLM | LogicLM+CAGE | LogicLM+Gated | SymbCoT | SymbCoT+CAGE | SymbCoT+Gated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ProofWriter | 0.11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| PrOntoQA | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FOLIO | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0098 | 0.00 | 0.00 |

Accepted-change audit:

| Dataset | Method | Accepted/changed | Correct | Wrong | Neutral | Net vs base |
|---|---|---:|---:|---:|---:|---:|
| ProofWriter-100 | Direct+Gated | 18 | 6 | 5 | 7 | +0.01 |
| ProofWriter-100 | LogicLM+Gated | 3 | 0 | 3 | 0 | -0.03 |
| ProofWriter-100 | SymbCoT+Gated | 8 | 4 | 4 | 0 | +0.00 |
| PrOntoQA-100 | Direct+Gated | 3 | 1 | 0 | 2 | +0.01 |
| PrOntoQA-100 | LogicLM+Gated | 3 | 0 | 3 | 0 | -0.03 |
| PrOntoQA-100 | SymbCoT+Gated | 14 | 11 | 3 | 0 | +0.08 |
| FOLIO-204 | Direct+Gated | 65 | 55 | 8 | 2 | +0.230 |
| FOLIO-204 | LogicLM+Gated | 14 | 11 | 3 | 0 | +0.039 |
| FOLIO-204 | SymbCoT+Gated | 47 | 39 | 4 | 4 | +0.172 |

## Counterfactual metrics summary

ProofWriter-100 and PrOntoQA-100 also have `results_counterfactual_llm.csv`, so report counterfactual metrics in addition to base accuracy. FOLIO does not have the same counterfactual result file in this experiment layout.

### Aggregate counterfactual metrics

| Dataset | Method | CF rows | CF Acc | CF consistency | Label-change acc | Irrelevant robustness | Attribution consistency |
|---|---|---:|---:|---:|---:|---:|---:|
| ProofWriter-100 | `llm_direct` | 200 | 0.580 | 0.650 | 0.759 | 0.453 | 0.352 |
| ProofWriter-100 | `llm_direct_cage` | 200 | 0.630 | 0.670 | 0.807 | 0.504 | 0.357 |
| ProofWriter-100 | `llm_direct_cage_gated` | 200 | 0.630 | 0.670 | 0.807 | 0.504 | 0.357 |
| ProofWriter-100 | `llm_logiclm` | 200 | 0.730 | 0.770 | 0.807 | 0.675 | 0.709 |
| ProofWriter-100 | `llm_logiclm_cage` | 200 | 0.735 | 0.770 | 0.819 | 0.675 | 0.709 |
| ProofWriter-100 | `llm_logiclm_cage_gated` | 200 | 0.735 | 0.770 | 0.819 | 0.675 | 0.709 |
| ProofWriter-100 | `llm_symbcot` | 200 | 0.450 | 0.590 | 0.325 | 0.538 | 0.352 |
| ProofWriter-100 | `llm_symbcot_cage` | 200 | 0.450 | 0.590 | 0.325 | 0.538 | 0.352 |
| ProofWriter-100 | `llm_symbcot_cage_gated` | 200 | 0.450 | 0.590 | 0.325 | 0.538 | 0.352 |
| PrOntoQA-100 | `llm_direct` | 200 | 0.605 | 0.670 | 0.780 | 0.430 | 0.500 |
| PrOntoQA-100 | `llm_direct_cage` | 200 | 0.615 | 0.675 | 0.790 | 0.440 | 0.505 |
| PrOntoQA-100 | `llm_direct_cage_gated` | 200 | 0.615 | 0.675 | 0.790 | 0.440 | 0.505 |
| PrOntoQA-100 | `llm_logiclm` | 200 | 0.785 | 0.825 | 0.870 | 0.700 | 0.785 |
| PrOntoQA-100 | `llm_logiclm_cage` | 200 | 0.785 | 0.825 | 0.870 | 0.700 | 0.785 |
| PrOntoQA-100 | `llm_logiclm_cage_gated` | 200 | 0.785 | 0.825 | 0.870 | 0.700 | 0.785 |
| PrOntoQA-100 | `llm_symbcot` | 200 | 0.255 | 0.495 | 0.010 | 0.500 | 0.500 |
| PrOntoQA-100 | `llm_symbcot_cage` | 200 | 0.255 | 0.495 | 0.010 | 0.500 | 0.500 |
| PrOntoQA-100 | `llm_symbcot_cage_gated` | 200 | 0.255 | 0.495 | 0.010 | 0.500 | 0.500 |

### Counterfactual family accuracy

| Dataset | Method | Proof-breaking | Proof-preserving | Alternate-proof | Contradiction injection |
|---|---|---:|---:|---:|---:|
| ProofWriter-100 | `llm_direct` | 0.707 | 0.500 | 0.000 | 0.250 |
| ProofWriter-100 | `llm_direct_cage` | 0.772 | 0.540 | 0.000 | 0.250 |
| ProofWriter-100 | `llm_direct_cage_gated` | 0.772 | 0.540 | 0.000 | 0.250 |
| ProofWriter-100 | `llm_logiclm` | 0.793 | 0.690 | 1.000 | 0.000 |
| ProofWriter-100 | `llm_logiclm_cage` | 0.804 | 0.690 | 1.000 | 0.000 |
| ProofWriter-100 | `llm_logiclm_cage_gated` | 0.804 | 0.690 | 1.000 | 0.000 |
| ProofWriter-100 | `llm_symbcot` | 0.380 | 0.530 | 0.250 | 0.250 |
| ProofWriter-100 | `llm_symbcot_cage` | 0.380 | 0.530 | 0.250 | 0.250 |
| ProofWriter-100 | `llm_symbcot_cage_gated` | 0.380 | 0.530 | 0.250 | 0.250 |
| PrOntoQA-100 | `llm_direct` | 0.780 | 0.430 | n/a | n/a |
| PrOntoQA-100 | `llm_direct_cage` | 0.790 | 0.440 | n/a | n/a |
| PrOntoQA-100 | `llm_direct_cage_gated` | 0.790 | 0.440 | n/a | n/a |
| PrOntoQA-100 | `llm_logiclm` | 0.870 | 0.700 | n/a | n/a |
| PrOntoQA-100 | `llm_logiclm_cage` | 0.870 | 0.700 | n/a | n/a |
| PrOntoQA-100 | `llm_logiclm_cage_gated` | 0.870 | 0.700 | n/a | n/a |
| PrOntoQA-100 | `llm_symbcot` | 0.010 | 0.500 | n/a | n/a |
| PrOntoQA-100 | `llm_symbcot_cage` | 0.010 | 0.500 | n/a | n/a |
| PrOntoQA-100 | `llm_symbcot_cage_gated` | 0.010 | 0.500 | n/a | n/a |

Counterfactual interpretation:

- Conservative CAGE improves Direct counterfactual behavior on ProofWriter and PrOntoQA, especially proof-breaking and proof-preserving accuracy.
- LogicLM already has the best counterfactual metrics; conservative/gated CAGE mostly preserve or slightly improve it.
- SymbCoT has weak counterfactual behavior on these generated counterfactuals, especially PrOntoQA proof-breaking, and CAGE wrappers do not improve the counterfactual-result file metrics there.
- Gated and conservative often have identical counterfactual metrics because this counterfactual evaluation is run on generated counterfactual examples separately, not only on base examples where gated repair changed the original answer.



### ProofWriter-100 and PrOntoQA-100

| Dataset | Method | Attr-P | Attr-R | Attr-F1 | Exact support |
|---|---|---:|---:|---:|---:|
| ProofWriter-100 | `llm_direct` | 0.000 | 0.000 | 0.000 | 0.000 |
| ProofWriter-100 | `llm_direct_cage` | 0.079 | 0.081 | 0.080 | 0.060 |
| ProofWriter-100 | `llm_direct_cage_gated` | 0.123 | 0.132 | 0.126 | 0.090 |
| ProofWriter-100 | `llm_logiclm` | 0.651 | 0.652 | 0.648 | 0.540 |
| ProofWriter-100 | `llm_logiclm_cage` | 0.651 | 0.652 | 0.648 | 0.540 |
| ProofWriter-100 | `llm_logiclm_cage_gated` | 0.656 | 0.660 | 0.655 | 0.550 |
| ProofWriter-100 | `llm_symbcot` | 0.000 | 0.000 | 0.000 | 0.000 |
| ProofWriter-100 | `llm_symbcot_cage` | 0.000 | 0.000 | 0.000 | 0.000 |
| ProofWriter-100 | `llm_symbcot_cage_gated` | 0.048 | 0.067 | 0.053 | 0.030 |
| PrOntoQA-100 | `llm_direct` | 0.000 | 0.000 | 0.000 | 0.000 |
| PrOntoQA-100 | `llm_direct_cage` | 0.018 | 0.018 | 0.018 | 0.010 |
| PrOntoQA-100 | `llm_direct_cage_gated` | 0.025 | 0.028 | 0.026 | 0.010 |
| PrOntoQA-100 | `llm_logiclm` | 0.732 | 0.697 | 0.697 | 0.400 |
| PrOntoQA-100 | `llm_logiclm_cage` | 0.732 | 0.697 | 0.697 | 0.400 |
| PrOntoQA-100 | `llm_logiclm_cage_gated` | 0.733 | 0.697 | 0.697 | 0.400 |
| PrOntoQA-100 | `llm_symbcot` | 0.000 | 0.000 | 0.000 | 0.000 |
| PrOntoQA-100 | `llm_symbcot_cage` | 0.000 | 0.000 | 0.000 | 0.000 |
| PrOntoQA-100 | `llm_symbcot_cage_gated` | 0.093 | 0.120 | 0.103 | 0.010 |

### FOLIO-204

FOLIO result files do not expose attribution metrics in the same way, so the useful non-accuracy audit there is the repair/change breakdown.

| Method | Changed | Valid changed | Improved | Hurt | Repair triggered | Repair accepted | Accepted precision | Unknown->det | Unknown->det wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `llm_direct_cage` | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 0 | 0 |
| `llm_direct_cage_gated` | 65 | 65 | 55 | 8 | 88 | 65 | 0.846 | 65 | 8 |
| `llm_logiclm_cage` | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 0 | 0 |
| `llm_logiclm_cage_gated` | 14 | 14 | 11 | 3 | 16 | 14 | 0.786 | 13 | 3 |
| `llm_symbcot_cage` | 2 | 0 | 2 | 0 | 2 | 2 | 1.000 | 0 | 0 |
| `llm_symbcot_cage_gated` | 47 | 45 | 39 | 4 | 57 | 47 | 0.830 | 26 | 4 |

Interpretation:

- ProofWriter and PrOntoQA provide attribution/support metrics, so those are the best places to compare diagnostic quality.
- Conservative wrappers generally preserve the base attribution profile, while gated wrappers slightly improve support metrics when they repair useful invalid or weak outputs.
- FOLIO does not expose the same attribution columns in its result file, so the main additional metrics there are repair acceptance and the unknown-overwrite audit.


- Conservative CAGE is usable as the main safe wrapper: it eliminates invalid outputs and preserves valid base labels. It is strongest as a non-degradation/diagnostic wrapper rather than a broad accuracy improver.
- Gated CAGE is usable as an enhanced/ablation variant, especially on FOLIO and PrOntoQA SymbCoT. It should not replace conservative CAGE as the main method because it hurts LogicLM on ProofWriter-100 and PrOntoQA-100.
- The FOLIO full result is strong and reportable: all three gated variants improve over their base methods, with Direct +23.0 points and SymbCoT +17.2 points.
- PrOntoQA-100 is mixed: SymbCoT+Gated improves +8 points, Direct+Gated is nearly flat, and LogicLM+Gated drops -3 points.
- ProofWriter-100 is not a gated success: Direct+Gated is worse than Direct+CAGE, LogicLM+Gated drops -3 points, and SymbCoT+Gated is flat.
- For paper tables, mark ProofWriter and PrOntoQA as 100-example runs unless rerun with the intended full dataset size. Do not describe them as full-dataset results.




- `outputs/smoke_proofwriter_20_method_cage_gated/`
- `outputs/smoke_prontoqa2_20_method_cage_gated/`
- `outputs/smoke_folio_20_method_cage_gated/`

Coverage and invalid checks:

- All three datasets have 20 rows for each of the nine method-agnostic comparison methods.
- Conservative wrappers preserve all valid base labels in all three datasets.
- Gated wrappers remove invalid Direct labels where present; final invalid rate is 0.00 for every gated method.

Accuracy comparison:

| Dataset | Direct | Direct+CAGE | Direct+Gated | LogicLM | LogicLM+CAGE | LogicLM+Gated | SymbCoT | SymbCoT+CAGE | SymbCoT+Gated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ProofWriter | 0.45 | 0.50 | 0.45 | 0.55 | 0.55 | 0.55 | 0.40 | 0.40 | 0.45 |
| PrOntoQA | 0.30 | 0.40 | 0.40 | 0.65 | 0.65 | 0.65 | 0.50 | 0.50 | 0.50 |
| FOLIO | 0.55 | 0.55 | 0.75 | 0.70 | 0.70 | 0.70 | 0.65 | 0.65 | 0.70 |

Accepted-change audit:

| Dataset | Method | Accepted changes | Correct | Wrong | Net accuracy delta vs base |
|---|---|---:|---:|---:|---:|
| ProofWriter | Direct+Gated | 5 | 2 | 2 | +0.00; one invalid repair was neutral |
| ProofWriter | LogicLM+Gated | 0 | 0 | 0 | +0.00 |
| ProofWriter | SymbCoT+Gated | 1 | 1 | 0 | +0.05 |
| PrOntoQA | Direct+Gated | 2 | 2 | 0 | +0.10 |
| PrOntoQA | LogicLM+Gated | 0 | 0 | 0 | +0.00 |
| PrOntoQA | SymbCoT+Gated | 0 | 0 | 0 | +0.00 |
| FOLIO | Direct+Gated | 8 | 6 | 2 | +0.20 |
| FOLIO | LogicLM+Gated | 2 | 1 | 1 | +0.00 |
| FOLIO | SymbCoT+Gated | 3 | 2 | 1 | +0.05 |

Important failure modes:

- FOLIO gated repair often helps Direct, but its wrong changes mostly overwrite gold `unknown` examples with `true` or `false`.
- ProofWriter Direct+Gated is not better than conservative Direct+CAGE: it repairs invalid labels but also changes two valid false labels to true incorrectly (`pw_pw_00016`, `pw_pw_00018`).
- LogicLM+Gated is mostly neutral; it rarely accepts repairs and does not improve over the base/conservative wrapper in these 20-example smokes.
- SymbCoT+Gated gives small gains on ProofWriter and FOLIO and is neutral on PrOntoQA.

Decision after 20-example smokes:

- Do **not** replace conservative CAGE with gated CAGE as the main method yet. Conservative remains the safer main wrapper because it preserves valid labels and fixes invalid Direct outputs.
- Keep gated CAGE as an enhanced/ablation variant. The result is promising enough to report because it improves Direct substantially on FOLIO and does not hurt PrOntoQA, but it is unstable on ProofWriter Direct and has an `unknown` overwrite failure mode.
- If running a 100-example follow-up, prioritize:
  1. Conservative nine-method comparison as the safe main result.
  2. Gated as an ablation/enhanced variant, with explicit reporting of accepted-repair precision and unknown-overwrite errors.
- Before making gated a main-table method, consider adding an unknown-protection gate: require stronger evidence before changing an initial `unknown` to `true`/`false`, or report separate metrics for `unknown -> determinate` accepted repairs.



Coverage check:

- The output contains 20 rows for each expected base/conservative/gated method.
- Additional legacy FOLIO methods also ran in this command/output: `folio_llm_fol`, `folio_llm_strategy`, `llm_cpa`, `llm_cage`, and `llm_cage_select`. They are not part of the nine-method method-agnostic CAGE comparison but can be used as side references.
- Invalid rate is 0.00 for all methods in this FOLIO smoke.

Accuracy summary:

| Pair | Base | Conservative | Gated | Gated delta vs base | Gated delta vs conservative |
|---|---:|---:|---:|---:|---:|
| Direct | 0.55 | 0.55 | 0.75 | +0.20 | +0.20 |
| LogicLM | 0.70 | 0.70 | 0.70 | +0.00 | +0.00 |
| SymbCoT | 0.65 | 0.65 | 0.70 | +0.05 | +0.05 |

Accepted repair / changed-label audit:

- `llm_direct_cage_gated`: 12 repair calls, 8 accepted label changes. Six accepted changes were correct and two were wrong, for a net +4/20 accuracy gain.
  - Correct accepted changes: `folio_1`, `folio_5`, `folio_6`, `folio_10`, `folio_12`, `folio_14`.
  - Wrong accepted changes: `folio_0`, `folio_4`; both changed gold `unknown` base predictions to `true`.
  - Repair calls that did not change the final label: `folio_2`, `folio_7`, `folio_8`, `folio_16`.
- `llm_logiclm_cage_gated`: 2 accepted label changes. One was correct and one was wrong, so net accuracy was unchanged.
  - Correct: `folio_5` (`unknown` -> `true`, gold `true`).
  - Wrong: `folio_7` (`unknown` -> `false`, gold `unknown`).
- `llm_symbcot_cage_gated`: 3 accepted label changes. Two were correct and one was wrong, for a net +1/20 accuracy gain.
  - Correct: `folio_6`, `folio_9` (`unknown` -> `true`, gold `true`).
  - Wrong: `folio_7` (`unknown` -> `false`, gold `unknown`).

Interpretation:

- FOLIO is a positive signal for gated repair: Direct gains substantially and SymbCoT gains slightly, while LogicLM stays neutral.
- The main failure mode is over-converting gold `unknown` examples into `true` or `false`, especially `folio_0`, `folio_4`, and `folio_7`.
- This suggests the gate is useful but still somewhat biased against preserving `unknown`. Before scaling gated to 100 as a main result, finish ProofWriter and PrOntoQA 20-example smokes. If the same unknown-overwrite pattern appears there, consider a stricter unknown-protection rule or report gated as an enhanced ablation.

## Paper result-integration plan

Use the current result set in the paper, but label dataset sizes accurately.

Must state clearly:

- FOLIO is a full validation-set run with 204 examples.
- ProofWriter and PrOntoQA current reported runs are 100-example runs, not full-dataset runs, unless rerun with full sizes.
- Conservative CAGE and Gated CAGE should be described as two variants:
  - **CAGE-Conservative:** safe wrapper; preserves valid base labels; repairs invalid labels; main method for safety/non-degradation.
  - **CAGE-Gated:** accuracy-seeking repair variant; accepts repair only under strong diagnostic agreement; best reported as enhanced/ablation variant.

Recommended paper tables:

1. **Main accuracy + invalid-rate table**
   - Include Base, Base+CAGE-Conservative, Base+CAGE-Gated for Direct, LogicLM, and SymbCoT.
   - Highlight strong improvements:
     - FOLIO Direct: 0.441 -> 0.672 (+23.0 points) with Gated.
     - FOLIO SymbCoT: 0.471 -> 0.642 (+17.2 points) with Gated.
     - FOLIO LogicLM: 0.637 -> 0.676 (+3.9 points) with Gated.
     - PrOntoQA-100 SymbCoT: 0.430 -> 0.510 (+8.0 points) with Gated.
     - ProofWriter-100 Direct: invalid rate 0.110 -> 0.000 and conservative accuracy 0.470 -> 0.510.
   - Also disclose weaker/mixed cases:
     - ProofWriter-100 LogicLM+Gated drops 0.720 -> 0.690.
     - PrOntoQA-100 LogicLM+Gated drops 0.680 -> 0.650.

2. **Counterfactual robustness table** for ProofWriter-100 and PrOntoQA-100
   - Report CF accuracy, counterfactual consistency, label-change accuracy, irrelevant robustness, and attribution consistency.
   - Highlight:
     - ProofWriter Direct CF Acc: 0.580 -> 0.630 with CAGE.
     - ProofWriter Direct label-change accuracy: 0.759 -> 0.807 with CAGE.
     - PrOntoQA Direct CF Acc: 0.605 -> 0.615 with CAGE.
     - LogicLM is strongest overall on counterfactual metrics: ProofWriter CF Acc 0.730/0.735 and PrOntoQA CF Acc 0.785.
   - Explain that gated and conservative are often identical on counterfactual files because counterfactual evaluation is run as separate examples, not only on original-example repairs.

3. **Attribution/support table** for ProofWriter-100 and PrOntoQA-100
   - Report attribution F1 and exact support match.
   - Highlight:
     - ProofWriter Direct support improves from Attr-F1 0.000 / exact 0.000 to Gated Attr-F1 0.126 / exact 0.090.
     - PrOntoQA SymbCoT support improves from Attr-F1 0.000 to Gated Attr-F1 0.103.
     - LogicLM remains the strongest support-tracking baseline: ProofWriter Attr-F1 about 0.648; PrOntoQA Attr-F1 about 0.697.

4. **Repair quality / ablation table**
   - Include repair triggered, accepted, accepted precision, changed valid labels, and unknown-overwrite errors.
   - Highlight strong accepted precision on FOLIO:
     - Direct+Gated accepted precision 0.846.
     - LogicLM+Gated accepted precision 0.786.
     - SymbCoT+Gated accepted precision 0.830.
   - Highlight PrOntoQA-100 SymbCoT+Gated accepted precision 0.786.
   - Disclose failures:
     - ProofWriter-100 Direct+Gated accepted precision 0.333.
     - ProofWriter/PrOntoQA LogicLM+Gated accepted precision 0.000 in these 100-example runs.
     - FOLIO unknown-overwrite errors: Direct 8, LogicLM 3, SymbCoT 4.

Recommended narrative:

- Use CAGE-Conservative as the main method when arguing safety and process-faithfulness: it preserves valid base answers, removes invalid outputs, and slightly improves counterfactual robustness for Direct.
- Use CAGE-Gated as evidence that diagnostic feedback can produce real accuracy gains when the gate is reliable, especially on FOLIO and SymbCoT-PrOntoQA.
- Do not claim Gated is universally better. Present it as a high-reward but higher-risk variant with explicit repair-quality diagnostics.
- Emphasize that the best results are not just accuracy improvements but also improvements in invalid-rate, attribution/support metrics, and counterfactual robustness.

## Figure/table presentation reference from AstroVisBench

Reference paper inspected: `C:\Users\cyf90\source\论文\2287_AstroVisBench_A_Code_Benc.pdf`.

Useful presentation patterns to borrow:

1. **Early overview figure**
   - AstroVisBench uses Figure 1 as a pipeline overview spanning task input -> model output -> execution/evaluation -> judge decision.
   - For our paper, use an early Figure 1 showing SymbCoT/LogicLM/Direct base reasoning -> CAGE diagnostics -> conservative/gated repair -> final label + support metrics.
   - The overview should be schematic and explanatory, not just a result plot.

2. **Compact multi-metric result table**
   - AstroVisBench Table 5 groups multiple evaluation dimensions in one compact table, with columns split into conceptual blocks.
   - For our main table, group columns as:
     - Answer quality: Accuracy, Invalid rate.
     - Faithfulness/support: Attr-F1, Exact support.
     - Counterfactual robustness: CF Acc, CF consistency.
     - Repair quality for gated: Trigger %, Accept %, Accepted precision.
   - Use short column names and a caption explaining each metric.

3. **Example grid figure**
   - AstroVisBench Figure 3 uses an eight-panel qualitative grid with labels (a)-(h) and a compact caption.
   - For our paper, use a small qualitative grid of CAGE cases:
     - successful invalid repair;
     - successful gated repair;
     - conservative preservation where aggressive repair would hurt;
     - unknown-overwrite failure case.
   - This helps make the diagnostic/repair mechanism concrete.

4. **Error-analysis figure**
   - AstroVisBench Figure 4 uses horizontal bar charts to summarize error types.
   - For our paper, use a bar chart for repair outcomes or failure modes:
     - correct accepted repair;
     - wrong accepted repair;
     - neutral repair;
     - unknown->determinate wrong;
     - valid label wrongly changed.
   - This is especially useful for explaining why gated is an ablation rather than the main method.

5. **Benchmark/statistics table style**
   - AstroVisBench Tables 2-4 are small side/inline tables that summarize supporting evidence without dominating the page.
   - For our paper, use small supporting tables for dataset sizes, method variants, and diagnostic metrics.

6. **Related-work comparison table**
   - AstroVisBench Table 6 compares prior benchmarks by Domain / Task Focus.
   - If space permits, include a compact related-work table comparing ordinary prompting, LogicLM/SymbCoT, verifier-style repair, and CAGE along axes like symbolic support, counterfactual diagnostics, conservative preservation, and gated repair.

Recommended layout for our result section:

- Figure 1: method overview pipeline.
- Table 1: dataset/method setup and evaluation metrics.
- Table 2: main answer quality table: accuracy + invalid rate for Base / Conservative / Gated.
- Table 3: faithfulness and counterfactual metrics: Attr-F1, Exact support, CF Acc, CF consistency.
- Table 4 or Figure 4: repair-quality ablation: trigger/accept/accepted precision/unknown-overwrite.
- Figure 2: qualitative examples grid.

Caption style to mimic:

- Captions should be long enough to define metrics and say what is being compared.
- Each table caption should explicitly state what columns mean, not rely on the main text only.
- Figures should use labeled subpanels `(a)`, `(b)`, etc., with a caption that explains the key takeaway.

Caution:

- Do not overcrowd the main table. If necessary, put detailed repair-quality and family-level counterfactual results in appendix, but keep the strongest FOLIO and counterfactual improvements in the main paper.

## LLM full-run dispatch fix

Issue found after attempting PrOntoQA-500 and ProofWriter-500:

- Non-FOLIO LLM runs were capped by `n = min(args.n_examples, 100)` inside `src/cf_reasoning/llm_client.py`, so `--n-examples 500` still loaded only 100 examples.
- `_llm_predictions()` always ran legacy `llm_cpa`, `llm_cage`, and `llm_cage_select` before the requested `--llm-baseline-methods`, causing unnecessary API calls and interruptions in old CAGE paths.

Fix applied:

- Removed the hard 100-example cap for non-FOLIO LLM runs.
- Added a baseline-only path: when `--llm-baseline-methods` is provided, run only the requested method-agnostic baselines/wrappers and skip legacy `llm_cpa`, `llm_cage`, and `llm_cage_select`.
- Kept old behavior when no baseline method list is provided.

Verification:

- `python -m compileall src/cf_reasoning/llm_client.py` passed.
- Monkeypatch sanity check confirmed `n_examples=500` reaches `_load_llm_examples` as 500 and dispatches only the baseline-only path.

Correct rerun commands:

```bash
python -m cf_reasoning.run_experiment --llm --dataset prontoqa --prontoqa-path data/raw/prontoqa2/ProntoQA_dev_gpt-4.json --n-examples 500 --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage,direct_cage_gated,logiclm_cage_gated,symbcot_cage_gated --output-dir outputs/full500_prontoqa2_method_cage_conservative_gated_fixed
```

```bash
python -m cf_reasoning.run_experiment --llm --dataset proofwriter --proofwriter-path data/raw/proofwriter/data-test.jsonl --n-examples 500 --llm-baseline-methods logiclm,symbcot,direct_cage,logiclm_cage,symbcot_cage,direct_cage_gated,logiclm_cage_gated,symbcot_cage_gated --output-dir outputs/proofwriter_500_method_cage_conservative_gated_fixed
```

## Final fixed large-run audit

Output directories:

- ProofWriter-500: `outputs/proofwriter_500_method_cage_conservative_gated_fixed/`
- PrOntoQA-499: `outputs/full500_prontoqa2_method_cage_conservative_gated_fixed/`
- FOLIO-204: `outputs/full_folio_method_cage_conservative_gated/`

Coverage:

- ProofWriter fixed run has 500 rows per requested method.
- PrOntoQA fixed run has 499 rows per requested method because 499 examples were parseable/usable.
- FOLIO has 204 rows per method and is the full validation set.
- Fixed non-FOLIO baseline-only runs intentionally omit standalone `llm_direct` rows in `results_llm.csv`; Direct base labels are available through wrapper traces and were used for inferred comparisons.

Main accuracy / invalid / support summary:

| Dataset | Method | N | Acc | Invalid | Attr-F1 | Exact support |
|---|---|---:|---:|---:|---:|---:|
| ProofWriter-500 | Direct inferred | 500 | 0.456 | 0.072 | n/a | n/a |
| ProofWriter-500 | `llm_direct_cage` | 500 | 0.492 | 0.000 | 0.057 | 0.040 |
| ProofWriter-500 | `llm_direct_cage_gated` | 500 | 0.492 | 0.000 | 0.095 | 0.068 |
| ProofWriter-500 | `llm_logiclm` | 500 | 0.770 | 0.006 | 0.657 | 0.524 |
| ProofWriter-500 | `llm_logiclm_cage` | 500 | 0.772 | 0.000 | 0.658 | 0.524 |
| ProofWriter-500 | `llm_logiclm_cage_gated` | 500 | 0.744 | 0.000 | 0.664 | 0.532 |
| ProofWriter-500 | `llm_symbcot` | 500 | 0.508 | 0.000 | 0.000 | 0.000 |
| ProofWriter-500 | `llm_symbcot_cage` | 500 | 0.508 | 0.000 | 0.000 | 0.000 |
| ProofWriter-500 | `llm_symbcot_cage_gated` | 500 | 0.528 | 0.000 | 0.064 | 0.040 |
| PrOntoQA-499 | Direct inferred | 499 | 0.449 | 0.024 | n/a | n/a |
| PrOntoQA-499 | `llm_direct_cage` | 499 | 0.461 | 0.000 | 0.020 | 0.004 |
| PrOntoQA-499 | `llm_direct_cage_gated` | 499 | 0.465 | 0.000 | 0.027 | 0.008 |
| PrOntoQA-499 | `llm_logiclm` | 499 | 0.627 | 0.000 | 0.678 | 0.361 |
| PrOntoQA-499 | `llm_logiclm_cage` | 499 | 0.627 | 0.000 | 0.678 | 0.361 |
| PrOntoQA-499 | `llm_logiclm_cage_gated` | 499 | 0.605 | 0.000 | 0.680 | 0.363 |
| PrOntoQA-499 | `llm_symbcot` | 499 | 0.491 | 0.000 | 0.000 | 0.000 |
| PrOntoQA-499 | `llm_symbcot_cage` | 499 | 0.491 | 0.000 | 0.000 | 0.000 |
| PrOntoQA-499 | `llm_symbcot_cage_gated` | 499 | 0.527 | 0.000 | 0.100 | 0.016 |
| FOLIO-204 | `llm_direct` | 204 | 0.441 | 0.000 | n/a | n/a |
| FOLIO-204 | `llm_direct_cage` | 204 | 0.441 | 0.000 | n/a | n/a |
| FOLIO-204 | `llm_direct_cage_gated` | 204 | 0.672 | 0.000 | n/a | n/a |
| FOLIO-204 | `llm_logiclm` | 204 | 0.637 | 0.000 | n/a | n/a |
| FOLIO-204 | `llm_logiclm_cage` | 204 | 0.637 | 0.000 | n/a | n/a |
| FOLIO-204 | `llm_logiclm_cage_gated` | 204 | 0.676 | 0.000 | n/a | n/a |
| FOLIO-204 | `llm_symbcot` | 204 | 0.471 | 0.010 | n/a | n/a |
| FOLIO-204 | `llm_symbcot_cage` | 204 | 0.480 | 0.000 | n/a | n/a |
| FOLIO-204 | `llm_symbcot_cage_gated` | 204 | 0.642 | 0.000 | n/a | n/a |

Counterfactual summary:

| Dataset | Method | CF rows | CF Acc | CF consistency | Label-change acc | Irrelevant robust | Attr consistency |
|---|---|---:|---:|---:|---:|---:|---:|
| ProofWriter-500 | `llm_direct_cage` | 1000 | 0.622 | 0.672 | 0.818 | 0.482 | 0.379 |
| ProofWriter-500 | `llm_direct_cage_gated` | 1000 | 0.622 | 0.672 | 0.818 | 0.482 | 0.378 |
| ProofWriter-500 | `llm_logiclm` | 1000 | 0.724 | 0.774 | 0.794 | 0.674 | 0.683 |
| ProofWriter-500 | `llm_logiclm_cage` | 1000 | 0.726 | 0.775 | 0.796 | 0.676 | 0.683 |
| ProofWriter-500 | `llm_logiclm_cage_gated` | 1000 | 0.726 | 0.775 | 0.796 | 0.676 | 0.683 |
| ProofWriter-500 | `llm_symbcot` | 1000 | 0.438 | 0.574 | 0.312 | 0.528 | 0.346 |
| ProofWriter-500 | `llm_symbcot_cage` | 1000 | 0.438 | 0.574 | 0.312 | 0.528 | 0.346 |
| ProofWriter-500 | `llm_symbcot_cage_gated` | 1000 | 0.438 | 0.574 | 0.312 | 0.528 | 0.346 |
| PrOntoQA-499 | `llm_direct_cage` | 998 | 0.595 | 0.655 | 0.742 | 0.449 | 0.503 |
| PrOntoQA-499 | `llm_direct_cage_gated` | 998 | 0.595 | 0.655 | 0.742 | 0.449 | 0.503 |
| PrOntoQA-499 | `llm_logiclm` | 998 | 0.804 | 0.830 | 0.896 | 0.711 | 0.771 |
| PrOntoQA-499 | `llm_logiclm_cage` | 998 | 0.804 | 0.830 | 0.896 | 0.711 | 0.771 |
| PrOntoQA-499 | `llm_logiclm_cage_gated` | 998 | 0.804 | 0.830 | 0.896 | 0.711 | 0.771 |
| PrOntoQA-499 | `llm_symbcot` | 998 | 0.261 | 0.512 | 0.012 | 0.509 | 0.500 |
| PrOntoQA-499 | `llm_symbcot_cage` | 998 | 0.261 | 0.512 | 0.012 | 0.509 | 0.500 |
| PrOntoQA-499 | `llm_symbcot_cage_gated` | 998 | 0.261 | 0.512 | 0.012 | 0.509 | 0.500 |

Repair-quality summary:

| Dataset | Method | Changed | Valid changed | Improved | Hurt | Triggered | Accepted | Accepted precision | Unknown->det wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ProofWriter-500 | `llm_direct_cage_gated` | 61 | 25 | 30 | 12 | 108 | 61 | 0.492 | 0 |
| ProofWriter-500 | `llm_logiclm_cage_gated` | 20 | 17 | 3 | 16 | 75 | 20 | 0.150 | 0 |
| ProofWriter-500 | `llm_symbcot_cage_gated` | 44 | 44 | 27 | 17 | 88 | 44 | 0.614 | 0 |
| PrOntoQA-499 | `llm_direct_cage_gated` | 16 | 4 | 9 | 1 | 31 | 16 | 0.562 | 0 |
| PrOntoQA-499 | `llm_logiclm_cage_gated` | 13 | 13 | 1 | 12 | 18 | 13 | 0.077 | 0 |
| PrOntoQA-499 | `llm_symbcot_cage_gated` | 66 | 66 | 41 | 23 | 86 | 66 | 0.621 | 0 |
| FOLIO-204 | `llm_direct_cage_gated` | 65 | 65 | 55 | 8 | 88 | 65 | 0.846 | 8 |
| FOLIO-204 | `llm_logiclm_cage_gated` | 14 | 14 | 11 | 3 | 16 | 14 | 0.786 | 3 |
| FOLIO-204 | `llm_symbcot_cage_gated` | 47 | 45 | 39 | 4 | 57 | 47 | 0.830 | 4 |

Final decision:

- Results are usable for the paper if labeled as ProofWriter-500, PrOntoQA-499, and FOLIO-204.
- CAGE-Conservative should be the main safe method: it eliminates invalid outputs and preserves valid labels.
- CAGE-Gated should be an enhanced/ablation variant: strong on FOLIO and useful for SymbCoT on ProofWriter/PrOntoQA, but harmful for LogicLM on non-FOLIO.
- Best headline results: FOLIO Direct+Gated 0.441 -> 0.672; FOLIO SymbCoT+Gated 0.471 -> 0.642; FOLIO LogicLM+Gated 0.637 -> 0.676; PrOntoQA SymbCoT+Gated 0.491 -> 0.527; ProofWriter SymbCoT+Gated 0.508 -> 0.528; ProofWriter Direct conservative invalid repair 0.072 -> 0.000 and accuracy 0.456 -> 0.492.
- Main caveat: LogicLM+Gated harms non-FOLIO results, so do not claim gated is universally better.

## Abstract revision following AstroVisBench style

Revised the abstract in `report/neurips2026_cage_draft.tex` to match the concise benchmark-paper structure observed in AstroVisBench:

1. Motivation and evaluation gap: final-label accuracy misses premise-level process faithfulness.
2. Proposed framework: CAGE with proof-breaking/proof-preserving counterfactual tests.
3. Scope: Direct, Logic-LM-style, and SymbCoT; conservative and gated variants.
4. Main findings: invalid-output elimination/preservation and strongest FOLIO gains.
5. Significance: unified reporting of answer correctness and process faithfulness.

The abstract now avoids implementation-level detail, long metric lists, and over-specific caveats while retaining the strongest defensible result numbers. It remains double-blind and contains no acknowledgement or identity information.


Updated `report/neurips2026_cage_draft.tex` after final fixed large-run audit.

Requirements addressed:

- Uses the NeurIPS 2026 template via `\usepackage{neurips_2026}`.
- Keeps double-blind author field: `\author{Anonymous Authors}`.
- No Acknowledgement section is present.
- Removed preliminary placeholders and replaced them with final result tables.
- Result wording labels dataset sizes accurately: ProofWriter-500, PrOntoQA-499, FOLIO-204 full validation.

Paper content updated:

- Abstract now summarizes final CAGE-Conservative and CAGE-Gated findings.
- Contributions now mention conservative and gated repair.
- Experiments section now defines datasets, variants, metrics, and provider-agnostic implementation.
- Results now include:
  - Main accuracy + invalid-rate table.
  - Support + counterfactual faithfulness table.
  - Repair-quality audit table.
- Discussion now frames Conservative as the safe main method and Gated as enhanced/ablation.
- Limitations now mention ProofWriter-500/PrOntoQA-499 scale, FOLIO transfer diagnostics, wrong-but-valid preservation, and gated over-repair.

Verification performed:

- Static scan found no `acknowledg`, `preliminary`, `todo`, local usernames, institution strings, or GitHub/release identity markers in the paper source.
- Local `pdflatex` is not installed in this shell (`pdflatex: command not found`), so final PDF page count still needs to be checked on a machine with LaTeX installed.

Next required check:

Compile with LaTeX and confirm the main text from abstract through conclusion is at most 7 pages, excluding references and appendix. If it exceeds 7 pages, first compress table font/spacing or move repair-quality details to appendix.
