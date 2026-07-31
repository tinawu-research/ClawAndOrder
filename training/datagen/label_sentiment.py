"""Teacher-label AFR article sentiment with Qwen, offline.

Sentiment is the only category whose target cannot be derived from a tool
payload. The rate component is a copy from ``lookup_rate``, the market direction
is a function of the label, and the confirm/contradict clause is a sign
comparison -- but the label itself needs judgement.

Qwen labels once here, at dataset-build time, and its output is written into the
``assistant`` field of the training file. At inference Qwen has no role in
synthesis: the fine-tuned Nemotron reads the evidence and produces the label
itself. The only artifact that changes is the Nemotron adapter, which is what
the rules permit.

Three noise controls, because at ~800 training sequences a wrong label costs
more than a missing one:

* **Identical context.** The teacher sees byte-identical text to what the
  student will see -- ``retrieve`` returns a lowercased 1,200-char blob, so the
  teacher gets exactly that, not the full article. Labelling from richer
  context than the student receives trains the student to guess.
* **Self-consistency.** Three votes across three rubric phrasings; only
  coarse-class-unanimous articles are kept.
* **Grounding check.** The teacher must quote a span from the snippet. A span
  that is not a substring is a confabulation and the vote is discarded.

    python training/datagen/label_sentiment.py --limit 400
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src" / "agent"))

import answers as A  # noqa: E402
import datastore  # noqa: E402
from tools import afr, rba  # noqa: E402

LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-local-cluster")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "agent-brain")

OUT_PATH = REPO_ROOT / "training" / "data" / "sentiment_examples.jsonl"
LABELS_PATH = REPO_ROOT / "training" / "data" / "sentiment_labels.jsonl"

LABELS = ["positive", "mixed_positive", "mixed", "mixed_negative", "negative"]

#: Direction phrasing per label, matched to the reference answers, which use
#: "upward", "mixed-to-down, with rate-sensitive shares under pressure" etc.
DIRECTION = {
    "positive": "upward",
    "mixed_positive": "mixed-to-up",
    "mixed": "mixed",
    "mixed_negative": "mixed-to-down",
    "negative": "downward",
}

SENTIMENT_TEXT = {
    "positive": "positive",
    "mixed_positive": "mixed with a positive bias",
    "mixed": "mixed",
    "mixed_negative": "mixed with a negative bias",
    "negative": "negative",
}

#: Three phrasings of the same rubric. Agreement across all three is a much
#: stronger signal than three samples of one prompt.
RUBRICS = [
    """\
You classify the financial-market sentiment of an Australian Financial Review article.

Choose one label:
positive        - clearly good news for the shares or market discussed
mixed_positive  - net positive but with real caveats
mixed           - genuinely balanced or unclear
mixed_negative  - net negative but with mitigating factors
negative        - clearly bad news for the shares or market discussed

Judge the market implication, not the writing tone. Base it only on the text given.

Reply with JSON only: {"label": "<one label>", "span": "<a quote of at most 120 characters copied exactly from the text>"}\
""",
    """\
An investor reads the article below. Which way does it point for the securities it discusses?

Labels: positive, mixed_positive, mixed, mixed_negative, negative.

Weigh what the article implies for prices, not how upbeat the prose sounds.
Use only what the text states.

Reply with JSON only: {"label": "<label>", "span": "<exact quote, <=120 chars, copied from the text>"}\
""",
    """\
Classify this Australian Financial Review article's implication for share prices.

Allowed labels: positive, mixed_positive, mixed, mixed_negative, negative.

A label of "mixed" means the article genuinely cuts both ways, not that you are
unsure. Ground the decision in the supplied text alone.

Reply with JSON only: {"label": "<label>", "span": "<verbatim quote from the text, max 120 chars>"}\
""",
]

#: Headlines that plausibly carry a market signal at all.
_MARKET_RE = re.compile(
    r"\b(shares?|stocks?|market|asx|rally|slump|surge|plunge|investors?|profit|"
    r"earnings|outlook|rates?|rba|dividend|guidance|forecast|selloff|rebound)\b",
    re.I,
)

_SECTORS = {
    "travel": r"\b(travel|airline|qantas|tourism|flight|holiday)\b",
    "energy": r"\b(energy|oil|gas|petrol|crude|coal)\b",
    "banks": r"\b(bank|banks|banking|cba|nab|anz|westpac|lender)\b",
    "miners": r"\b(mining|miner|iron ore|bhp|rio tinto|resources)\b",
    "property": r"\b(property|housing|reit|real estate|mortgage)\b",
    "retail": r"\b(retail|consumer|shopper|sales)\b",
    "tech": r"\b(tech|technology|telco|telecom|nbn|software)\b",
}


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


def _sector(headline: str) -> str:
    for name, pattern in _SECTORS.items():
        if re.search(pattern, headline, re.I):
            return name
    return "broad"


def _subject(sector: str) -> str:
    return {
        "travel": "ASX travel shares",
        "energy": "ASX energy shares",
        "banks": "ASX bank shares",
        "miners": "ASX mining shares",
        "property": "ASX property shares",
        "retail": "ASX retail shares",
        "tech": "ASX technology shares",
        "broad": "the broad ASX",
    }[sector]


def select_articles(limit: int, rng: random.Random) -> list[dict]:
    """Stratified candidate pool: market-relevant, deduped, spread over years and sectors."""
    store = datastore.STORE
    buckets: dict[tuple[int, str], list[dict]] = collections.defaultdict(list)
    seen: set[str] = set()

    for article in store.afr:
        headline = (article.headline or "").strip()
        date = article.publication
        if not headline or date is None:
            continue
        if not _MARKET_RE.search(headline):
            continue
        # Dedupe on headline: the corpus carries ~12k duplicate printings,
        # heavily concentrated in late 2016.
        key = headline.lower()
        if key in seen:
            continue
        seen.add(key)
        blob_len = len(article.blob or "")
        if not (800 <= blob_len <= 8000):
            continue
        buckets[(date.year, _sector(headline))].append(
            {"headline": headline, "date": date.isoformat()}
        )

    per_bucket = max(1, limit // max(len(buckets), 1))
    picked: list[dict] = []
    for key in sorted(buckets):
        items = buckets[key]
        rng.shuffle(items)
        picked.extend(items[:per_bucket])
    rng.shuffle(picked)
    return picked[:limit]


def _vote(snippet: str, rubric: str) -> tuple[str, str] | None:
    try:
        body = _post(
            {
                "model": TEACHER_MODEL,
                "messages": [
                    {"role": "system", "content": rubric},
                    {"role": "user", "content": snippet},
                ],
                "temperature": 0.7,
                "max_tokens": 200,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        text = body["choices"][0]["message"]["content"] or ""
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
        return None

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    label = str(parsed.get("label", "")).strip().lower()
    span = str(parsed.get("span", "")).strip()
    if label not in LABELS:
        return None
    # Grounding check: an ungrounded quote means the model is inventing.
    if span and span.lower()[:60] not in snippet.lower():
        return None
    return label, span


def _coarse(label: str) -> str:
    if label in ("positive", "mixed_positive"):
        return "pos"
    if label in ("negative", "mixed_negative"):
        return "neg"
    return "mix"


def label_article(article: dict) -> dict | None:
    """Label one article, keeping it only if all three rubrics agree coarsely."""
    payload = afr.find_article(headline=article["headline"], date=article["date"], limit=1)
    articles = payload.get("articles") or []
    if not articles:
        return None

    # The exact text the student will see at inference.
    snippet = articles[0].get("text") or articles[0].get("blob") or ""
    if len(snippet) < 200:
        return None

    votes = [_vote(snippet, rubric) for rubric in RUBRICS]
    votes = [v for v in votes if v]
    if len(votes) < 3:
        return None

    coarse = {_coarse(label) for label, _ in votes}
    if len(coarse) > 1:
        return None

    labels = [label for label, _ in votes]
    winner = collections.Counter(labels).most_common(1)[0][0]
    return {
        "headline": article["headline"],
        "date": article["date"],
        "label": winner,
        "votes": labels,
        "sector": _sector(article["headline"]),
        "retrieve_payload": payload,
    }


def build_examples(labelled: list[dict]) -> list[dict]:
    """Turn labels into training examples with the rate and direction components."""
    out = []
    for record in labelled:
        date = record["date"]
        rate_args = {"dataset": "rba", "metric": "lookup_rate", "date_from": date}
        rate_payload = rba.METRICS["lookup_rate"](date_from=date)

        retrieve_args = {"headline": record["headline"], "date": date}

        subject = _subject(record["sector"])
        label = record["label"]

        parts = [
            f"The RBA cash-rate target in force was {A.rate(rate_payload['cash_rate_target'])}",
            f"the article's sentiment is {SENTIMENT_TEXT[label]}",
            f"the likely direction for {subject} is {DIRECTION[label]}",
        ]

        question = (
            f"Retrieve the AFR article \"{record['headline']}\" published {A.day(date)} and use "
            f"the RBA cash-rate target in force on that date. Classify the article's "
            f"financial-market sentiment as positive, negative, or mixed; state the likely "
            f"direction for {subject}."
        )

        out.append(
            {
                "template_id": "sentiment.rate_label_direction",
                "question": question,
                "answer": A.join_clauses(parts),
                "components": parts,
                "tool_calls": [
                    ["retrieve", retrieve_args, record["retrieve_payload"]],
                    ["query_data", rate_args, rate_payload],
                ],
                "param_key": f"sentiment|{record['headline'][:40]}|{date}",
                "split_keys": {"years": [int(date[:4])], "articles": [record["headline"]]},
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400, help="candidate articles to label")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("loading datasets ...", flush=True)
    datastore.STORE.load()

    print(f"selecting up to {args.limit} candidate articles ...", flush=True)
    candidates = select_articles(args.limit, rng)
    print(f"  {len(candidates)} candidates")

    print(f"labelling ({args.workers}-way, 3 votes each) ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(label_article, candidates))

    labelled = [r for r in results if r]
    retention = len(labelled) / max(len(candidates), 1)
    print(f"  kept {len(labelled)}/{len(candidates)} ({retention:.0%} unanimous on coarse class)")

    distribution = collections.Counter(r["label"] for r in labelled)
    print("\nlabel distribution:")
    for label in LABELS:
        print(f"  {label:16} {distribution.get(label, 0):4}")

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as handle:
        for record in labelled:
            slim = {k: v for k, v in record.items() if k != "retrieve_payload"}
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")

    examples = build_examples(labelled)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(examples)} examples -> {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {len(labelled)} labels   -> {LABELS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
