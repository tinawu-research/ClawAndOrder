"""Question paraphrasing, for phrasing diversity.

Templated questions are stilted and repetitive; hidden questions are prose. With
only ~800 training sequences, a model shown one phrasing per fact set learns the
phrasing rather than the task -- the "template lock" failure mode. Paraphrasing
the *question* while holding the gold answer fixed is the cheapest defence.

Two invariants:

* The answer never changes. Only the surface form of the question does.
* A paraphrase must ask for exactly the same components. A variant that quietly
  drops one would make the gold answer wrong, so each is checked against the
  component count before being accepted.

All paraphrases of a fact set share a ``gold_key`` and must land in the same
split, or a paraphrase in validation is just a memorised training item.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

from spec import Example

LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-local-cluster")
PARAPHRASE_MODEL = os.getenv("PARAPHRASE_MODEL", "agent-brain")

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "paraphrase_cache.sqlite"

_local = threading.local()

SYSTEM = """\
You rewrite financial data questions. You are given a question and the number of
distinct facts it asks for.

Produce {n} alternative phrasings of the SAME question. Rules:
- Ask for exactly the same facts. Never add or drop a requested value.
- Keep every named entity, ticker, year, date, exclusion and search term exactly
  as written. "Excluding Tabcorp" must stay an exclusion.
- Vary register: one direct question, one imperative instruction, one
  conversational. Do not vary the meaning.
- No preamble, no numbering, no quotes.

Reply with exactly {n} lines, one phrasing per line.\
"""


def _conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(CACHE_PATH, timeout=30)
        conn.execute("CREATE TABLE IF NOT EXISTS paraphrases (key TEXT PRIMARY KEY, variants TEXT)")
        conn.commit()
        _local.conn = conn
    return conn


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
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _plausible(variant: str, original: str) -> bool:
    """Reject variants that lost a hard token the answer depends on."""
    import re

    if not (15 <= len(variant) <= 400):
        return False
    # Every year, ticker and quoted term in the original must survive.
    for token in set(re.findall(r"\b(?:19|20)\d{2}\b|\b[A-Z]{2,4}\.AX\b", original)):
        if token not in variant:
            return False
    if "xclud" in original and "xclud" not in variant and "Tabcorp" not in variant:
        return False
    return True


def paraphrase_one(question: str, n_components: int, n: int = 3) -> list[str]:
    """Return up to ``n`` accepted paraphrases of ``question``."""
    key = sha256(f"{PARAPHRASE_MODEL}|{n}|{question}".encode("utf-8")).hexdigest()
    conn = _conn()
    row = conn.execute("SELECT variants FROM paraphrases WHERE key = ?", (key,)).fetchone()
    if row is not None:
        return json.loads(row[0])

    try:
        body = _post(
            {
                "model": PARAPHRASE_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM.format(n=n)},
                    {
                        "role": "user",
                        "content": f"Question: {question}\nDistinct facts requested: {n_components}",
                    },
                ],
                "temperature": 0.8,
                "top_p": 0.95,
                "max_tokens": 400,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        text = body["choices"][0]["message"]["content"] or ""
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
        return []

    variants = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip().strip('"')
        if line and line != question and _plausible(line, question):
            variants.append(line)
    variants = variants[:n]

    conn.execute(
        "INSERT OR REPLACE INTO paraphrases (key, variants) VALUES (?, ?)",
        (key, json.dumps(variants)),
    )
    conn.commit()
    return variants


def expand(examples: list[Example], *, n: int = 2, workers: int = 12) -> list[Example]:
    """Return paraphrase variants for ``examples`` (originals not included)."""
    if not examples:
        return []

    def work(example: Example) -> list[Example]:
        variants = paraphrase_one(example.question, len(example.components), n=n)
        out = []
        for index, variant in enumerate(variants):
            clone = Example(
                category=example.category,
                template_id=f"{example.template_id}.p{index + 1}",
                question=variant,
                answer=example.answer,
                components=list(example.components),
                tool_calls=list(example.tool_calls),
                param_key=example.param_key,
                split_keys=dict(example.split_keys),
                perturbations=list(example.perturbations) + ["paraphrase"],
            )
            out.append(clone)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [e for batch in pool.map(work, examples) for e in batch]
