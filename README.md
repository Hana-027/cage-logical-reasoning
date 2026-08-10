# CAGE Logical Reasoning

This repository contains the code for **Counterfactual Attribution-Guided Repair for Process-Faithful Logical Reasoning in Large Language Models**.

- Paper: [OpenReview](https://openreview.net/forum?id=jh5KPtvfdf)

## Overview

CAGE evaluates and repairs logical-reasoning outputs by combining final-answer accuracy with process-faithfulness diagnostics. The framework probes whether a predicted answer is causally supported by the premises through proof-breaking, proof-preserving, contradiction, support-shift, and verifier-style interventions.

The repository includes:

- deterministic logical-reasoning baselines;
- counterfactual generation and evaluation utilities;
- CAGE conservative and gated repair policies;
- adapters for Direct, LogicLM-style, SymbCoT-style, and FOLIO runs;
- tests for loaders, metrics, counterfactuals, repair logic, and LLM parsing.

## Repository layout

```text
configs/              Default experiment configuration
src/cf_reasoning/     Python package implementation
tests/                Unit tests
```

Large datasets, generated outputs, external baseline repositories, caches, and local credentials are intentionally excluded from version control via `.gitignore`.

## Installation

Use Python 3.10 or later.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows PowerShell, activate with:

```bash
.venv/Scripts/Activate.ps1
```

Install the package in editable mode:

```bash
pip install -e .
```

For tests:

```bash
pip install -e ".[test]"
```

For LLM-backed experiments:

```bash
pip install -e ".[llm]"
```

## Running tests

```bash
python -m pytest
```

The current public release was checked with the local test suite before upload.

## Example usage

Run deterministic offline baselines on synthetic data:

```bash
python -m cf_reasoning.run_experiment --offline --dataset synthetic --n-examples 100 --output-dir outputs/smoke
```

Run on a local ProofWriter directory:

```bash
python -m cf_reasoning.run_experiment --offline --dataset proofwriter --proofwriter-root path/to/proofwriter --max-per-split 100 --output-dir outputs/proofwriter
```

Run on local FOLIO validation data:

```bash
python -m cf_reasoning.run_experiment --llm --dataset folio --folio-path path/to/folio-validation.jsonl --output-dir outputs/folio
```

## LLM configuration

LLM runs use a provider-agnostic interface. Configure credentials through environment variables rather than committing keys.

For DeepSeek:

```bash
export CF_REASONING_LLM_PROVIDER=deepseek
```

```bash
export DEEPSEEK_API_KEY=your_key_here
```

Optional:

```bash
export DEEPSEEK_MODEL=deepseek-chat
```

Common optional settings:

```bash
export CF_REASONING_MAX_TOKENS=512
```

```bash
export CF_REASONING_TEMPERATURE=0
```

Do not commit `.env` files or API keys.

## Data

This repository does not include full raw datasets or generated experiment outputs. Place datasets under local paths such as `data/raw/` and pass their locations through CLI arguments, for example `--proofwriter-root`, `--prontoqa-path`, or `--folio-path`.

## Citation

If you use this code or build on the method, please cite the paper:

```bibtex
@misc{cage2026,
  title = {Counterfactual Attribution-Guided Repair for Process-Faithful Logical Reasoning in Large Language Models},
  author = {Anonymous Authors},
  year = {2026},
  url = {https://openreview.net/forum?id=jh5KPtvfdf}
}
```

## License

No license has been specified yet. Until a license is added, all rights are reserved by default.
