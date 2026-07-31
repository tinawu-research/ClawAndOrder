"""Shared types for generated training examples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any

from render import build_messages, make_trace_entry


@dataclass
class Example:
    """One generated training example.

    ``components`` is the list of facts the question asked for, in the order the
    answer states them. It exists so the builder can self-grade the example and
    so the split logic can detect near-duplicates -- it is not shown to the
    model.
    """

    category: str
    template_id: str
    question: str
    answer: str
    components: list[str]
    tool_calls: list[tuple[str, dict[str, Any], Any]]
    param_key: str = ""
    split_keys: dict[str, Any] = field(default_factory=dict)
    perturbations: list[str] = field(default_factory=list)
    tokens: int = 0

    @property
    def gold_key(self) -> str:
        """Identity of the underlying fact set.

        Paraphrases and surface variants of the same question share this, and
        must therefore land in the same split.
        """
        payload = f"{self.category}|{self.param_key}|{'|'.join(sorted(self.components))}"
        return sha1(payload.encode("utf-8")).hexdigest()[:16]

    def trace(self) -> list[dict[str, Any]]:
        return [make_trace_entry(tool, args, payload) for tool, args, payload in self.tool_calls]

    def messages(self) -> list[dict[str, str]]:
        return build_messages(self.question, self.trace(), self.answer)

    def to_record(self) -> dict[str, Any]:
        return {
            "messages": self.messages(),
            "meta": {
                "category": self.category,
                "template_id": self.template_id,
                "param_key": self.param_key,
                "gold_key": self.gold_key,
                "components": self.components,
                "split_keys": self.split_keys,
                "perturbations": self.perturbations,
                "tokens": self.tokens,
            },
        }

    def fingerprint(self) -> str:
        """Content hash used for cross-split collision detection."""
        return sha1(
            json.dumps(self.messages(), sort_keys=True).encode("utf-8")
        ).hexdigest()
