"""Token counting against the real Nemotron tokenizer.

Character-count heuristics are not good enough here. Supervised fine-tuning
truncates from the right, so an example that overruns ``max_seq_len`` keeps its
system prompt and loses its assistant span entirely -- a training row with no
label, produced silently. With only ~800 sequences in the run, a handful of
those is a measurable share of the signal.

Counting goes through the served model's ``/tokenize`` endpoint, which uses the
exact tokenizer the trainer will use and avoids adding a transformers
dependency to the host environment. Results are memoised because the builder
tokenises every candidate example, often repeatedly across rebuilds.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from hashlib import sha256
from pathlib import Path

TOKENIZE_URL = os.getenv("TOKENIZE_URL", "http://10.0.1.11:8001/tokenize")
TOKENIZE_MODEL = os.getenv("TOKENIZE_MODEL", "nemotron-8b-finance")

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "token_cache.sqlite"

_local = threading.local()
_lock = threading.Lock()


class TokenizerError(RuntimeError):
    """The tokenizer endpoint could not be reached."""


def _conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(CACHE_PATH, timeout=30)
        conn.execute("CREATE TABLE IF NOT EXISTS tokens (key TEXT PRIMARY KEY, n INTEGER)")
        conn.commit()
        _local.conn = conn
    return conn


def count(text: str) -> int:
    """Number of tokens in ``text``."""
    key = sha256(f"{TOKENIZE_MODEL}\x00{text}".encode("utf-8")).hexdigest()
    conn = _conn()
    row = conn.execute("SELECT n FROM tokens WHERE key = ?", (key,)).fetchone()
    if row is not None:
        return int(row[0])

    request = urllib.request.Request(
        TOKENIZE_URL,
        method="POST",
        data=json.dumps({"model": TOKENIZE_MODEL, "prompt": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            n = int(json.load(response)["count"])
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
        raise TokenizerError(f"tokenizer unreachable at {TOKENIZE_URL}: {exc}") from exc

    conn.execute("INSERT OR REPLACE INTO tokens (key, n) VALUES (?, ?)", (key, n))
    conn.commit()
    return n


def count_messages(messages: list[dict[str, str]]) -> int:
    """Approximate the tokenised length of a full chat turn list.

    Adds a per-message allowance for the chat template's role headers and turn
    delimiters, which ``/tokenize`` does not apply to a raw prompt string. The
    allowance is deliberately generous: over-estimating costs a slightly
    tighter budget, under-estimating costs silently truncated labels.
    """
    total = sum(count(m["content"]) for m in messages)
    return total + 4 * len(messages) + 8


def fits(messages: list[dict[str, str]], budget: int) -> bool:
    return count_messages(messages) <= budget
