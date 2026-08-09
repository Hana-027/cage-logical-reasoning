from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class FolioExample:
    id: str
    premises: list[str]
    premises_fol: list[str]
    conclusion: str
    conclusion_fol: str
    label: str
    split: str = "folio"

    @property
    def text(self) -> str:
        lines = [f"P{i + 1}: {premise}" for i, premise in enumerate(self.premises)]
        lines.append(f"Conclusion: {self.conclusion}")
        return "\n".join(lines)


def load_folio_examples(path: str | Path, limit: int | None = None, split: str = "folio") -> list[FolioExample]:
    examples: list[FolioExample] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if limit is not None and len(examples) >= limit:
                break
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            label = _normalize_label(str(row.get("label", "")))
            if label not in {"true", "false", "unknown"}:
                continue
            examples.append(
                FolioExample(
                    id=f"folio_{row.get('example_id') or row_index}",
                    premises=[str(x) for x in row.get("premises", [])],
                    premises_fol=[str(x) for x in row.get("premises-FOL", [])],
                    conclusion=str(row.get("conclusion", "")),
                    conclusion_fol=str(row.get("conclusion-FOL", "")),
                    label=label,
                    split=split,
                )
            )
    return examples


def _normalize_label(label: str) -> str:
    label = label.strip().lower()
    if label in {"true", "entailment"}:
        return "true"
    if label in {"false", "contradiction"}:
        return "false"
    if label in {"unknown", "uncertain", "neutral"}:
        return "unknown"
    return label
