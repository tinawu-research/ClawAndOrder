"""Replay frozen evidence through a model arm — the base-vs-fine-tuned engine.

The comparison this feeds is 30% of the score, so the measurement has to isolate
synthesis. Running the live agent per arm would not: Qwen re-plans every call,
so the two arms would see different evidence and the delta would be a mix of
routing noise and synthesis quality.

Instead the tool traces are captured once (``freeze_evidence.py``) and replayed
byte-identically to every arm. Qwen routing and tools are held fixed, which is
what the handout asks for, and a per-arm sweep drops from ~30s to ~3s a question.

Four conditions, because a clean-only result only measures the best case:

``clean``
    The captured trace, untouched. Headline synthesis quality.
``noisy``
    Plus an irrelevant coverage block. Does the arm answer what was asked, or
    narrate everything it was handed?
``insufficient``
    One component's evidence removed. Does it state the limitation, or invent a
    figure? The rules require the former and the handout scores invention at zero.
``shuffled``
    Block order reversed and payload keys reordered. Did it learn to *read* the
    evidence, or to copy from a fixed position?

Arms are model ids on one vLLM process (``base``, ``ck20``, …), so switching arm
is a different ``model`` field, not a restart — identical hardware state between
measurements.

    python training/eval/replay.py --arms base ck20 --conditions clean
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "training" / "datagen"))

from render import build_messages  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src" / "agent"))
from compact import compact_payload  # noqa: E402

FROZEN_DIR = REPO_ROOT / "training" / "eval_sets" / "frozen_evidence"

#: Direct to the fine-tuning node by default. LiteLLM only exposes the two
#: aliases; the per-checkpoint arms exist as model ids on vLLM itself.
FT_URL = os.getenv("FT_URL", "http://10.0.1.11:8001/v1")
FT_KEY = os.getenv("LITELLM_KEY", "sk-local-cluster")
BASE_MODEL_ID = os.getenv("FT_BASE_MODEL_ID", "nemotron-8b-finance")

_TIMEOUT = int(os.getenv("REPLAY_TIMEOUT_SECONDS", "120"))
_MAX_RETRIES = 3

CONDITIONS = ("clean", "noisy", "insufficient", "shuffled")


@dataclass
class Generation:
    """One arm's answer to one question under one condition."""

    qid: str
    arm: str
    condition: str
    answer: str
    latency_seconds: float
    served_model: str = ""
    long_prompt: bool = False
    error: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        record = {
            "qid": self.qid,
            "arm": self.arm,
            "condition": self.condition,
            "answer": self.answer,
            "latency_seconds": round(self.latency_seconds, 3),
            "served_model": self.served_model,
            "long_prompt": self.long_prompt,
        }
        if self.error:
            record["error"] = self.error
        return record


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------

#: An off-topic block, in the shape the orchestrator actually emits. Deliberately
#: real evidence about the wrong thing rather than nonsense — a block the arm
#: could plausibly weave in is the harder distractor.
_DISTRACTOR = {
    "tool": "dataset_coverage",
    "args": {"dataset": "afr"},
    "result": json.dumps(
        {"dataset": "afr", "articles": 3891, "first": "2015-01-02", "last": "2021-12-31"}
    ),
    "compact": "dataset=afr articles=3891 first=2015-01-02 last=2021-12-31",
    "ok": True,
}


def _reorder_payload_keys(entry: dict[str, Any]) -> dict[str, Any]:
    """Reverse key order in the payload, leaving values untouched.

    ``compact`` must be re-rendered from the reordered payload, not merely
    carried over. ``format_evidence`` prefers ``compact`` and falls back to
    ``result``, so rewriting only the JSON would leave the model reading the
    original ordering and make this condition a silent no-op.
    """
    out = dict(entry)
    try:
        payload = json.loads(entry["result"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return out
    if not isinstance(payload, dict):
        return out
    payload = {k: payload[k] for k in reversed(list(payload))}
    out["result"] = json.dumps(payload, ensure_ascii=False)
    out["compact"] = compact_payload(payload)
    return out


def apply_condition(trace: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    """Derive a perturbed trace. Never mutates the input."""
    trace = copy.deepcopy(trace)
    if condition == "clean":
        return trace
    if condition == "noisy":
        return [copy.deepcopy(_DISTRACTOR), *trace]
    if condition == "insufficient":
        # Drop the last block. On a single-block trace this becomes the empty
        # trace, which is the real runtime path when planning yields nothing —
        # the case the N2 negatives were built for.
        return trace[:-1]
    if condition == "shuffled":
        return [_reorder_payload_keys(e) for e in reversed(trace)]
    raise ValueError(f"unknown condition: {condition}")


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {FT_KEY}"},
    )
    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            last = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code < 500:
                break                      # a 4xx will not fix itself on retry
        except Exception as exc:           # noqa: BLE001
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"replay request failed: {last}")


def generate(
    record: dict[str, Any],
    arm: str,
    condition: str,
    *,
    long_prompt: bool = False,
    max_tokens: int = 512,
) -> Generation:
    """Synthesise one answer from frozen evidence."""
    trace = apply_condition(record["tool_trace"], condition)
    messages = build_messages(record["prompt"], trace, long_prompt=long_prompt)

    payload: dict[str, Any] = {
        "model": arm,
        "messages": messages,
        # Greedy. A temperature here would put sampling noise inside a paired
        # comparison whose whole point is that the arms differ only in weights.
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": max_tokens,
    }

    started = time.monotonic()
    try:
        response = _post(f"{FT_URL}/chat/completions", payload)
    except Exception as exc:  # noqa: BLE001
        return Generation(
            qid=record["qid"],
            arm=arm,
            condition=condition,
            answer="",
            latency_seconds=time.monotonic() - started,
            long_prompt=long_prompt,
            error=str(exc),
            trace=trace,
        )
    elapsed = time.monotonic() - started

    answer = (response["choices"][0]["message"].get("content") or "").strip()
    return Generation(
        qid=record["qid"],
        arm=arm,
        condition=condition,
        answer=strip_reasoning(answer),
        latency_seconds=elapsed,
        # vLLM echoes the id it actually served. Captured per request because it
        # is the end-to-end receipt that the adapter, not the base, answered.
        served_model=response.get("model", ""),
        long_prompt=long_prompt,
        trace=trace,
    )


def strip_reasoning(text: str) -> str:
    """Remove a ``<think>…</think>`` preamble.

    Nemotron-Nano is a toggleable reasoning model. Left in reasoning mode the
    base arm buries its answer in commentary and grades badly for a reason that
    has nothing to do with fine-tuning, so both arms are stripped identically.
    """
    lowered = text.lower()
    start = lowered.find("<think>")
    if start == -1:
        return text.strip()
    end = lowered.find("</think>", start)
    if end == -1:
        return text[:start].strip()
    return (text[:start] + text[end + len("</think>"):]).strip()


def replay_arm(
    records: list[dict[str, Any]],
    arm: str,
    condition: str,
    *,
    long_prompt: bool = False,
    workers: int = 4,
) -> list[Generation]:
    """Generate every answer for one (arm, condition) cell.

    Concurrency is modest on purpose: the numbers reported alongside these
    answers include synthesis latency, and 16-way load on a single GB10 would
    make that latency a measure of queueing rather than of the model.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(generate, record, arm, condition, long_prompt=long_prompt)
            for record in records
        ]
        return [future.result() for future in futures]


def load_frozen(directory: Path = FROZEN_DIR, condition: str = "clean") -> list[dict[str, Any]]:
    """Load frozen traces. ``condition`` selects the capture variant suffix."""
    records = []
    for path in sorted(directory.glob(f"*.{condition}.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise SystemExit(f"no frozen evidence in {directory} (run freeze_evidence.py)")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["base"], help="model ids to sweep")
    parser.add_argument("--conditions", nargs="+", default=["clean"], choices=CONDITIONS)
    parser.add_argument("--long-prompt", action="store_true", help="pre-fine-tune system prompt")
    parser.add_argument("--out", default="training/results/generations.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    records = load_frozen()
    print(f"{len(records)} frozen questions from {FROZEN_DIR.relative_to(REPO_ROOT)}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generations: list[Generation] = []
    for arm in args.arms:
        # "base" is a convenience alias for whatever id vLLM serves the
        # unmodified weights under; every other arm is a LoRA module name.
        arm_id = BASE_MODEL_ID if arm == "base" else arm
        for condition in args.conditions:
            started = time.monotonic()
            batch = replay_arm(
                records, arm_id, condition,
                long_prompt=args.long_prompt, workers=args.workers,
            )
            for generation in batch:
                generation.arm = arm
            errors = sum(1 for g in batch if g.error)
            empty = sum(1 for g in batch if not g.error and not g.answer)
            latencies = sorted(g.latency_seconds for g in batch if not g.error)
            p50 = latencies[len(latencies) // 2] if latencies else float("nan")
            print(
                f"  {arm:8} {condition:13} n={len(batch):3} "
                f"errors={errors} empty={empty} p50={p50:5.1f}s "
                f"wall={time.monotonic() - started:5.1f}s"
            )
            if errors:
                first = next(g for g in batch if g.error)
                print(f"      first error ({first.qid}): {first.error[:200]}")
            generations.extend(batch)

    with open(out_path, "w", encoding="utf-8") as handle:
        for generation in generations:
            handle.write(json.dumps(generation.to_record(), ensure_ascii=False) + "\n")
    print(f"\nwrote {len(generations)} generations -> {args.out}")

    served = {g.served_model for g in generations if g.served_model}
    print(f"served model ids seen: {sorted(served)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
