"""Component-based LLM judge, replicating the organizers' grading protocol.

The official harness shows an LLM judge (Qwen3.6-35B, the same model served
behind ``agent-brain``) the question, the candidate answer, and **one expected
fact at a time**, and reads a YES/NO verdict. This module reproduces that.

Two implementation details matter more than they look:

``enable_thinking`` must be off
    Qwen3.6 emits a "Thinking Process:" preamble by default and will burn the
    whole ``max_tokens`` budget on it, returning no verdict. Passing
    ``chat_template_kwargs={"enable_thinking": False}`` is what makes an 8-token
    completion viable. Confirmed to pass through LiteLLM.

The verdict comes from logprobs, not the text
    Reading ``top_logprobs`` on the first content token gives a deterministic
    argmax over YES/NO plus a free calibration signal (``p_yes``). Parsing the
    string would throw that away and tie-break arbitrarily on stray whitespace.

Results are cached in sqlite keyed by the full input, so re-running an
evaluation over unchanged answers costs zero model calls.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

#: Bump to deliberately invalidate every cached verdict.
JUDGE_PROMPT_VERSION = "v1"

LITELLM_URL = os.getenv("JUDGE_LITELLM_URL", os.getenv("LITELLM_URL", "http://localhost:4000/v1"))
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-local-cluster")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "agent-brain")

CACHE_PATH = Path(__file__).resolve().parent.parent / "results" / "judge_cache.sqlite"

_TIMEOUT = int(os.getenv("JUDGE_TIMEOUT_SECONDS", "90"))
_MAX_RETRIES = 3


# The hedging and wrong-context clauses are not padding: with a bare "reply
# YES or NO" prompt the judge accepts "approximately 4.75" for the fact "4.75",
# which the scoring handout explicitly lists as a zero.
JUDGE_SYSTEM_PROMPT = """\
You are a strict grading judge for a financial question-answering benchmark.
You receive a QUESTION, a candidate ANSWER, and ONE EXPECTED FACT.
Decide whether the ANSWER states that expected fact.

Reply YES only if the answer asserts the expected fact.

Accept:
- equivalent numeric formatting: 1,234 == 1234; 2.50 == 2.5; +22.17% == "a 22.17 per cent gain"
- equivalent date formats: "Jan 2024" == "2024-01"; "2 Jan 2015" == "2015-01-02"
- minor rephrasing or synonyms that preserve the meaning
- values inside the tolerance stated in TOLERANCE, where that tolerance applies

Reply NO if:
- the value is absent, contradicted, or replaced by a different value
- the value is correct but stated about the wrong entity, period or quantity
- the value is hedged ("approximately 41", "roughly 20", "about 4.75")
- the answer refuses, or redirects to a different question
- the fact is only implied and never stated

If the EXPECTED FACT contains several values, ALL of them must be present and
correct for YES.

TOLERANCE: {tolerance_note}

Reply with exactly one word: YES or NO.\
"""

JUDGE_USER_TEMPLATE = """\
QUESTION: {question}

ANSWER:
{answer}

EXPECTED FACT: {expected_fact}

Does the answer state this expected fact? Reply YES or NO.\
"""


@dataclass(frozen=True)
class Verdict:
    """One graded component."""

    yes: bool
    p_yes: float
    raw: str
    cached: bool = False

    @property
    def confident(self) -> bool:
        """True when the judge was not close to the decision boundary."""
        return abs(self.p_yes - 0.5) > 0.35


class JudgeError(RuntimeError):
    """The judge could not be reached or returned an unreadable response."""


class _Cache:
    """Thread-safe sqlite verdict cache.

    One connection per thread: sqlite objects cannot cross threads, and the
    judge runs on a thread pool.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn().execute(
            "CREATE TABLE IF NOT EXISTS verdicts ("
            "  key TEXT PRIMARY KEY, yes INTEGER, p_yes REAL, raw TEXT"
            ")"
        )
        self._conn().commit()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=30)
            self._local.conn = conn
        return conn

    def get(self, key: str) -> Verdict | None:
        row = self._conn().execute(
            "SELECT yes, p_yes, raw FROM verdicts WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return Verdict(yes=bool(row[0]), p_yes=row[1], raw=row[2], cached=True)

    def put(self, key: str, verdict: Verdict) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO verdicts (key, yes, p_yes, raw) VALUES (?, ?, ?, ?)",
            (key, int(verdict.yes), verdict.p_yes, verdict.raw),
        )
        conn.commit()


_cache: _Cache | None = None
_cache_lock = threading.Lock()


def _get_cache() -> _Cache:
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = _Cache(CACHE_PATH)
    return _cache


def _cache_key(question: str, answer: str, expected_fact: str, tolerance_note: str) -> str:
    parts = [
        JUDGE_MODEL,
        JUDGE_PROMPT_VERSION,
        question,
        answer,
        expected_fact,
        tolerance_note,
    ]
    return sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _post(payload: dict) -> dict:
    request = urllib.request.Request(
        f"{LITELLM_URL.rstrip('/')}/chat/completions",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {LITELLM_KEY}",
            "Content-Type": "application/json",
        },
    )
    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
    raise JudgeError(f"judge unreachable after {_MAX_RETRIES} attempts: {last}")


def _verdict_from_logprobs(choice: dict) -> Verdict:
    """Read YES/NO off the first content token's top_logprobs.

    Falls back to string parsing when the server omits logprobs, so the harness
    still functions (with a degraded p_yes) rather than failing outright.
    """
    text = (choice.get("message", {}).get("content") or "").strip()
    logprobs = (choice.get("logprobs") or {}).get("content") or []

    if not logprobs:
        upper = text.upper()
        yes = upper.startswith("YES")
        return Verdict(yes=yes, p_yes=1.0 if yes else 0.0, raw=text)

    yes_lp: float | None = None
    no_lp: float | None = None
    for candidate in logprobs[0].get("top_logprobs", []):
        token = candidate.get("token", "").strip().upper()
        if token == "YES" and yes_lp is None:
            yes_lp = candidate["logprob"]
        elif token == "NO" and no_lp is None:
            no_lp = candidate["logprob"]

    if yes_lp is None and no_lp is None:
        upper = text.upper()
        yes = upper.startswith("YES")
        return Verdict(yes=yes, p_yes=1.0 if yes else 0.0, raw=text)

    # Renormalise over just the two verdict tokens; other candidates in the
    # top-k are formatting variants we do not want diluting the probability.
    yes_p = math.exp(yes_lp) if yes_lp is not None else 0.0
    no_p = math.exp(no_lp) if no_lp is not None else 0.0
    total = yes_p + no_p
    p_yes = yes_p / total if total > 0 else (1.0 if yes_lp is not None else 0.0)
    return Verdict(yes=p_yes >= 0.5, p_yes=p_yes, raw=text)


def judge_component(
    question: str,
    answer: str,
    expected_fact: str,
    tolerance_note: str = "",
    *,
    use_cache: bool = True,
) -> Verdict:
    """Grade one expected fact against one answer."""
    key = _cache_key(question, answer, expected_fact, tolerance_note)
    cache = _get_cache()
    if use_cache:
        hit = cache.get(key)
        if hit is not None:
            return hit

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT.format(
                    tolerance_note=tolerance_note or "Exact values are required."
                ),
            },
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    question=question, answer=answer, expected_fact=expected_fact
                ),
            },
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": 8,
        "logprobs": True,
        "top_logprobs": 5,
        # Without this Qwen3.6 spends the entire budget on a thinking preamble
        # and never emits a verdict.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    body = _post(payload)
    try:
        verdict = _verdict_from_logprobs(body["choices"][0])
    except (KeyError, IndexError) as exc:
        raise JudgeError(f"unexpected judge response: {json.dumps(body)[:400]}") from exc

    cache.put(key, verdict)
    return verdict


def judge_many(
    items: list[tuple[str, str, str, str]],
    *,
    workers: int = 16,
    use_cache: bool = True,
) -> list[Verdict]:
    """Grade many ``(question, answer, expected_fact, tolerance_note)`` tuples.

    Order is preserved. Runs on a thread pool because the work is HTTP-bound.
    """
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(
                lambda item: judge_component(*item, use_cache=use_cache),
                items,
            )
        )
