"""Evidence perturbations and negative examples.

Two distinct jobs.

**Perturbation** closes the gap between the tidy evidence a generator emits and
the messy evidence Qwen actually produces: a ``dataset_coverage`` probe it was
told to run first, an extra call it decided it wanted, the same metric fetched
twice with slightly different arguments, blocks in an arbitrary order. A model
trained only on minimal canonical traces learns to read position rather than
content.

**Negative examples** teach the behaviour the rules require: *"Return a
response for every question, even when evidence is insufficient. State the
limitation clearly in the answer field instead of returning an empty response
or inventing a figure."* That is answer-with-limitation, never bare refusal --
the handout scores a refusal at zero.

The discrimination guard matters as much as the negatives themselves. Without
paired complete-evidence examples of the same question shape, the model learns
"this kind of question means hedge", which converts full marks into none.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import answers as A
from spec import Example

#: Cheap, always-available blocks that make plausible distractors.
_DISTRACTOR_SPECS = [
    ("dataset_coverage", {}),
    ("query_data", {"dataset": "rba", "metric": "coverage"}),
    ("query_data", {"dataset": "afr", "metric": "coverage"}),
]


def _clone(example: Example, **overrides) -> Example:
    data = {
        "category": example.category,
        "template_id": example.template_id,
        "question": example.question,
        "answer": example.answer,
        "components": list(example.components),
        "tool_calls": list(example.tool_calls),
        "param_key": example.param_key,
        "split_keys": dict(example.split_keys),
        "perturbations": list(example.perturbations),
    }
    data.update(overrides)
    return Example(**data)


def add_coverage_probe(example: Example, coverage_call: tuple, rng: random.Random) -> Example:
    """Prepend a coverage call, which the brain's system prompt tells it to run."""
    return _clone(
        example,
        tool_calls=[coverage_call] + list(example.tool_calls),
        perturbations=example.perturbations + ["coverage_probe"],
    )


def add_distractor(example: Example, distractor: tuple, rng: random.Random) -> Example:
    """Insert one correct but irrelevant block at a random position."""
    calls = list(example.tool_calls)
    calls.insert(rng.randrange(len(calls) + 1), distractor)
    return _clone(
        example,
        tool_calls=calls,
        perturbations=example.perturbations + ["distractor"],
    )


def shuffle_blocks(example: Example, rng: random.Random) -> Example:
    """Permute block order. The answer must not depend on it."""
    if len(example.tool_calls) < 2:
        return example
    calls = list(example.tool_calls)
    rng.shuffle(calls)
    return _clone(
        example,
        tool_calls=calls,
        perturbations=example.perturbations + ["shuffled"],
    )


def duplicate_with_variant(example: Example, rng: random.Random) -> Example:
    """Repeat a call without its exclusion, so two blocks disagree.

    Qwen really does this -- it fetches a metric with and without
    ``exclude_tickers`` and leaves both in the trace. The model has to pick the
    block matching the question's stated exclusion.
    """
    candidates = [
        i for i, (tool, args, _) in enumerate(example.tool_calls)
        if tool == "query_data" and args.get("exclude_tickers")
    ]
    if not candidates:
        return example

    index = rng.choice(candidates)
    tool, args, payload = example.tool_calls[index]
    variant_args = {k: v for k, v in args.items() if k != "exclude_tickers"}

    # Re-run the metric without the exclusion so the duplicate block carries
    # genuinely different numbers rather than a copy.
    import generators  # local import: generators imports this module's siblings

    try:
        kwargs = {k: v for k, v in variant_args.items() if k not in ("dataset", "metric")}
        variant = generators._qd(args["dataset"], args["metric"], **kwargs)
    except Exception:
        return example

    calls = list(example.tool_calls)
    calls.insert(index, variant)
    return _clone(
        example,
        tool_calls=calls,
        perturbations=example.perturbations + ["contradictory_duplicate"],
    )


def truncate_block(example: Example, rng: random.Random) -> Example:
    """Mark one block as truncated, matching the orchestrator's overflow tail."""
    calls = list(example.tool_calls)
    index = rng.randrange(len(calls))
    tool, args, payload = calls[index]
    if not isinstance(payload, dict):
        return example
    clipped = copy.deepcopy(payload)
    clipped["_truncated"] = True
    calls[index] = (tool, args, clipped)
    return _clone(
        example,
        tool_calls=calls,
        perturbations=example.perturbations + ["truncated"],
    )


# --------------------------------------------------------------------------
# Negative examples
# --------------------------------------------------------------------------

_MISSING_PHRASES = [
    "The supplied data does not include {}",
    "The evidence provided does not cover {}",
    "{} is not supported by the supplied evidence",
]

_WHAT_IS_MISSING = {
    "asx_returns": "the requested ASX return",
    "asx_volume": "the requested average volume",
    "asx_drawdown": "the requested drawdown detail",
    "afr_counts": "the requested AFR count",
    "rba_counts": "the requested RBA figure",
    "rba_asx_event": "the requested ASX window return",
    "composite": "the requested ASX basket return",
    "coverage": "the requested coverage detail",
}


def drop_component(example: Example, rng: random.Random) -> Example | None:
    """Remove one tool block and the component it supported.

    The answer keeps every component the remaining evidence supports and then
    states the gap plainly, which is exactly the behaviour the rules mandate.
    """
    if len(example.tool_calls) < 2 or len(example.components) < 2:
        return None

    calls = list(example.tool_calls)
    dropped_index = rng.randrange(len(calls))
    calls.pop(dropped_index)

    kept = example.components[:-1]
    missing = _WHAT_IS_MISSING.get(example.category, "the remaining requested value")
    phrase = rng.choice(_MISSING_PHRASES).format(missing)

    return _clone(
        example,
        answer=A.join_clauses(kept + [phrase]),
        components=kept + [phrase],
        tool_calls=calls,
        template_id=example.template_id + ".n1_missing",
        perturbations=example.perturbations + ["n1_missing_component"],
    )


def empty_trace(example: Example, rng: random.Random) -> Example:
    """No evidence at all -- a real runtime path when planning fails.

    ``format_evidence`` returns a fixed string for an empty trace, and two of
    the fifteen public questions hit exactly this during trace capture when the
    brain timed out. The answer must still be a response, not a blank.
    """
    return _clone(
        example,
        answer=(
            "The supplied data does not contain the evidence needed to answer this "
            "question, so no grounded answer can be given."
        ),
        components=["no grounded answer can be given"],
        tool_calls=[],
        template_id=example.template_id + ".n2_empty",
        perturbations=example.perturbations + ["n2_empty_trace"],
    )


def apply_perturbations(
    examples: list[Example],
    rng: random.Random,
    *,
    rate: float = 0.35,
) -> list[Example]:
    """Perturb a fraction of ``examples`` in place of their clean versions."""
    import generators

    coverage_calls = []
    for tool, args in _DISTRACTOR_SPECS:
        try:
            if tool == "dataset_coverage":
                continue
            coverage_calls.append(
                generators._qd(args["dataset"], args["metric"])
            )
        except Exception:
            continue

    out: list[Example] = []
    for example in examples:
        if rng.random() >= rate or not example.tool_calls:
            out.append(example)
            continue

        choice = rng.random()
        if choice < 0.30 and coverage_calls:
            out.append(add_coverage_probe(example, rng.choice(coverage_calls), rng))
        elif choice < 0.55 and coverage_calls:
            out.append(add_distractor(example, rng.choice(coverage_calls), rng))
        elif choice < 0.75:
            out.append(shuffle_blocks(example, rng))
        elif choice < 0.90:
            out.append(duplicate_with_variant(example, rng))
        else:
            out.append(truncate_block(example, rng))
    return out


def build_negatives(
    examples: list[Example],
    rng: random.Random,
    *,
    target: int,
) -> list[Example]:
    """Build negative examples plus their paired complete-evidence controls.

    Returns roughly ``target`` negatives. The caller keeps the originals, which
    are the discrimination guard: same question shapes, full evidence, complete
    confident answers.
    """
    pool = [e for e in examples if len(e.tool_calls) >= 2 and len(e.components) >= 2]
    rng.shuffle(pool)

    negatives: list[Example] = []
    for example in pool:
        if len(negatives) >= target:
            break
        made = drop_component(example, rng)
        if made is not None:
            negatives.append(made)

    # A small slice of empty-trace cases; this path is real but rare, and
    # over-weighting it teaches the model to give up.
    n_empty = max(1, target // 8)
    for example in rng.sample(examples, min(n_empty, len(examples))):
        negatives.append(empty_trace(example, rng))

    return negatives
