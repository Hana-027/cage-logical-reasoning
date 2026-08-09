from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal as TypingLiteral

Label = TypingLiteral["true", "false", "unknown", "ambiguous"]
ExpectedRelation = TypingLiteral["changed", "preserved"]
CounterfactualFamily = TypingLiteral[
    "proof_breaking",
    "proof_preserving",
    "support_shift",
    "alternate_proof",
    "contradiction_injection",
    "distractor_injection",
    "paraphrase_preserving",
    "entity_swap",
    "rule_structure_intervention",
    "human_hard",
]
ProofRelation = TypingLiteral["broken", "preserved_same_support", "preserved_new_support", "shifted_support", "conflicting_support"]
DiagnosticDimension = TypingLiteral[
    "proof_causality",
    "proof_preservation",
    "support_shift_awareness",
    "conflict_handling",
    "distractor_robustness",
    "semantic_invariance",
    "entity_binding",
    "rule_understanding",
    "minimality",
]


@dataclass(frozen=True, order=True)
class Literal:
    predicate: str
    entity: str
    negated: bool = False

    def negate(self) -> "Literal":
        return Literal(self.predicate, self.entity, not self.negated)

    def to_text(self) -> str:
        name = self.entity.capitalize()
        if self.negated:
            return f"{name} is not {self.predicate}."
        return f"{name} is {self.predicate}."


@dataclass(frozen=True)
class Fact:
    id: str
    literal: Literal

    def to_text(self) -> str:
        return f"{self.id}: {self.literal.to_text()}"


@dataclass(frozen=True)
class Rule:
    id: str
    antecedents: tuple[Literal, ...]
    consequent: Literal

    def to_text(self) -> str:
        lhs = " and ".join(l.to_text().removesuffix(".").replace(l.entity.capitalize(), "someone") for l in self.antecedents)
        rhs = self.consequent.to_text().removesuffix(".").replace(self.consequent.entity.capitalize(), "they")
        return f"{self.id}: If {lhs}, then {rhs}."


@dataclass
class ProofResult:
    label: Label
    support_ids: tuple[str, ...]
    depth: int
    entailed: dict[Literal, tuple[frozenset[str], int]]


@dataclass
class Example:
    id: str
    facts: list[Fact]
    rules: list[Rule]
    query: Literal
    label: Label
    support_ids: tuple[str, ...]
    depth: int
    text: str
    split: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualExample:
    id: str
    source_id: str
    intervention_type: str
    changed_ids: tuple[str, ...]
    facts: list[Fact]
    rules: list[Rule]
    query: Literal
    label: Label
    support_ids: tuple[str, ...]
    depth: int
    expected_relation: ExpectedRelation
    text: str
    split: str = ""
    bundle_id: str = ""
    parent_id: str = ""
    cf_family: str = ""
    diagnostic_dimension: str = ""
    proof_relation: str = ""
    source_support_ids: tuple[str, ...] = ()
    target_support_ids: tuple[str, ...] = ()
    removed_support_ids: tuple[str, ...] = ()
    added_support_ids: tuple[str, ...] = ()
    support_overlap: float = 0.0
    edit_distance: int = 0
    is_minimal: bool = False
    conflict_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    example_id: str
    method: str
    label: str
    premise_ids: tuple[str, ...] = ()
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
