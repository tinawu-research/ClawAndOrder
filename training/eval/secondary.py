"""Deterministic answer metrics — no judge calls, no model, no cost.

The component score says whether the requested facts arrived. It does not say
*how* they arrived, and the failure modes fine-tuning is supposed to fix mostly
live in the "how": a base model that hedges every figure, narrates its
reasoning, or fills an unsupported slot with a plausible-looking number can
still score respectably on components while being unusable.

These metrics are computed from the answer and the frozen evidence alone, so
they are exactly reproducible and can be reported without a caveat about judge
variance. ``hallucinated_number_rate`` in particular is only computable because
the evidence is frozen — with live traces there would be nothing to compare a
numeric literal against.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Iterable

#: A hedge only matters next to a number: "approximately 41" is a scored zero,
#: while "the data broadly covers 2015-2021" is ordinary prose. The window is
#: what keeps this from flagging every discursive sentence.
_HEDGE_WORDS = (
    "approximately", "roughly", "about", "around", "circa", "some",
    "nearly", "almost", "just over", "just under", "in the region of",
    "or so", "give or take", "ballpark", "estimated", "an estimated",
    "close to", "upwards of", "north of", "south of",
)
_HEDGE_WINDOW = 40

_HEDGE_RE = re.compile(
    r"(?:%s)" % "|".join(re.escape(w) for w in _HEDGE_WORDS), re.IGNORECASE
)

#: Reasoning that leaked into a graded field. Nemotron-Nano is a toggleable
#: reasoning model, so this is a live risk on the base arm specifically.
_LEAK_RE = re.compile(
    r"<think>|</think>|thinking process|let me (?:think|check|work|see)|"
    r"^\s*(?:step \d|first,? i|okay,? so|hmm)|i need to (?:find|check|look)|"
    r"the user (?:is asking|wants)|based on the tool results?,? i",
    re.IGNORECASE | re.MULTILINE,
)

#: Config, tool plumbing, or scaffolding that must never reach ``answer``.
_FORMAT_VIOLATION_RE = re.compile(
    r"DOMAIN_PREDICT_MODE|mock-fallback|tool_trace|query_data\(|dataset_coverage\(|"
    r"\[\d+\]\s+(?:query_data|retrieve|dataset_coverage)|```|"
    r"^\s*\{\s*[\"']|Verified tool results",
    re.IGNORECASE | re.MULTILINE,
)

#: Numbers with a unit, a decimal, or a thousands separator. Bare small integers
#: are excluded by _is_scaffolding below rather than here, so that a genuine
#: count like "41" is still checked.
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass
class SecondaryMetrics:
    """Per-answer deterministic signals."""

    qid: str
    arm: str
    condition: str
    hedged: bool
    leaked_reasoning: bool
    format_violation: bool
    hallucinated_numbers: list[str]
    answer_chars: int
    answer_words: int
    empty: bool

    @property
    def hallucinated(self) -> bool:
        return bool(self.hallucinated_numbers)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["hallucinated"] = self.hallucinated
        return record


def _normalise_number(token: str) -> str:
    """Canonical form for comparison: no separators, no sign, no trailing zeros."""
    token = token.replace(",", "").lstrip("+-")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token or "0"


def evidence_numbers(trace: Iterable[dict[str, Any]]) -> set[str]:
    """Every numeric literal appearing anywhere in the evidence.

    Both surfaces are scanned. ``compact`` is what the model reads and
    ``result`` is the JSON the organizers see; a number present in either is
    grounded, and rounding differences between the two are why both are needed.
    """
    found: set[str] = set()
    for entry in trace:
        for field_name in ("compact", "result"):
            value = entry.get(field_name)
            if not value:
                continue
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            for token in _NUMBER_RE.findall(text):
                found.add(_normalise_number(token))
        # Argument values are evidence too: an excluded ticker or a top_n is
        # legitimately quotable in the answer.
        for token in _NUMBER_RE.findall(json.dumps(entry.get("args", {}), default=str)):
            found.add(_normalise_number(token))
    return found


def _rounding_variants(token: str) -> set[str]:
    """Values a correct answer may legitimately print for a grounded number.

    Evidence carries ``-82.4531``; the reference answer says ``-82.45%``. Both
    are the same fact, and counting the rounded form as hallucinated would make
    this metric fire hardest on the arms that format best.
    """
    variants = {token}
    try:
        value = float(token)
    except ValueError:
        return variants
    for places in (0, 1, 2, 3, 4):
        variants.add(_normalise_number(f"{value:.{places}f}"))
    return variants


def _is_scaffolding(token: str, raw: str, answer: str) -> bool:
    """List markers and enumerations, not claims about the data.

    ``1) AMP.AX …; 2) AGL.AX …`` is the required ranking format, so its ordinals
    must not read as invented figures.
    """
    if token in {"1", "2", "3", "4", "5"}:
        marker = re.search(rf"(?:^|[;\s]){re.escape(raw)}[).:]\s", answer, re.MULTILINE)
        if marker:
            return True
    return False


def hallucinated_numbers(
    answer: str,
    trace: Iterable[dict[str, Any]],
    question: str = "",
) -> list[str]:
    """Numeric literals asserted in the answer that no evidence supports.

    Numbers echoed from the question are grounded by definition — a question
    naming 2019 or "the three worst" licenses those figures in the reply.
    """
    grounded: set[str] = set()
    for token in evidence_numbers(trace):
        grounded |= _rounding_variants(token)
    for token in _NUMBER_RE.findall(question):
        grounded |= _rounding_variants(_normalise_number(token))

    unsupported: list[str] = []
    for raw in _NUMBER_RE.findall(answer):
        token = _normalise_number(raw)
        if token in grounded or _is_scaffolding(token, raw, answer):
            continue
        unsupported.append(raw)
    return unsupported


def hedge_flag(answer: str) -> bool:
    """True when a hedge word sits within 40 characters of a number."""
    for match in _HEDGE_RE.finditer(answer):
        window = answer[match.end(): match.end() + _HEDGE_WINDOW]
        if _NUMBER_RE.search(window):
            return True
    return False


def measure(
    qid: str,
    arm: str,
    condition: str,
    answer: str,
    trace: Iterable[dict[str, Any]],
    question: str = "",
) -> SecondaryMetrics:
    """All deterministic signals for one answer."""
    return SecondaryMetrics(
        qid=qid,
        arm=arm,
        condition=condition,
        hedged=hedge_flag(answer),
        leaked_reasoning=bool(_LEAK_RE.search(answer)),
        format_violation=bool(_FORMAT_VIOLATION_RE.search(answer)),
        hallucinated_numbers=hallucinated_numbers(answer, trace, question),
        answer_chars=len(answer),
        answer_words=len(answer.split()),
        empty=not answer.strip(),
    )


def summarise(metrics: list[SecondaryMetrics]) -> dict[str, Any]:
    """Rates over one (arm, condition) cell."""
    n = len(metrics)
    if not n:
        return {"n": 0}
    lengths = sorted(m.answer_words for m in metrics)
    return {
        "n": n,
        "hedge_rate": sum(m.hedged for m in metrics) / n,
        "hallucinated_number_rate": sum(m.hallucinated for m in metrics) / n,
        "leaked_reasoning_rate": sum(m.leaked_reasoning for m in metrics) / n,
        "format_violation_rate": sum(m.format_violation for m in metrics) / n,
        "empty_rate": sum(m.empty for m in metrics) / n,
        "answer_words_p50": lengths[n // 2],
        "answer_words_max": lengths[-1],
    }
