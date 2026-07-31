"""FastAPI service exposing the agent's evaluation contract.

    GET  /health   200 once the datasets are loaded. A hard gate: if this does
                   not return 200 at the start of the eval run, the team is
                   skipped entirely, so it reports 503 until the corpus is
                   actually queryable rather than lying early.
    POST /query    {"question": "..."} -> {"answer", "steps", "tool_trace"}

Plus, for humans rather than the harness:

    GET  /            operations dashboard (static, self-contained)
    GET  /api/status  dashboard telemetry
    GET  /api/public-questions   the 15 calibration cases
    POST /api/score   component-level self-scoring of one answer

Run it on all interfaces — the harness calls from another machine::

    uvicorn server:app --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from datastore import STORE
from orchestrator import answer_question

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("agent.server")

FRONTEND_DIR = Path(__file__).parent / "frontend"
PUBLIC_QUESTIONS = (
    Path(__file__).resolve().parents[2]
    / "Participant_Package"
    / "public_questions.jsonl"
)

_STARTED_AT = time.time()
#: Rolling per-request telemetry for the dashboard. Bounded, in-memory only.
_RECENT: list[dict[str, Any]] = []
_RECENT_LIMIT = 50
_RECENT_LOCK = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the corpus before accepting traffic.

    Runs in a worker thread so the event loop stays responsive and /health can
    answer 503-with-a-reason during the ~25s warm-up instead of hanging.
    """
    logger.info("loading datasets from %s", config.DATA_ROOT)
    task = asyncio.create_task(asyncio.to_thread(STORE.load))
    if not config.synthesis_is_live():
        logger.warning(
            "DOMAIN_PREDICT_MODE=%s — final synthesis is NOT using the "
            "fine-tuned model. Set DOMAIN_PREDICT_MODE=llm before official "
            "evaluation.",
            config.DOMAIN_PREDICT_MODE,
        )
    yield
    task.cancel()


app = FastAPI(
    title="ClawAndOrder market-signal agent",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


@app.get("/health")
async def health() -> JSONResponse:
    """Readiness gate for the evaluation harness."""
    if STORE.error:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": STORE.error},
        )
    if not STORE.ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "detail": "datasets still loading"},
        )
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.post("/query")
async def query(request: QueryRequest) -> dict[str, Any]:
    """Answer one question.

    Returns 503 only when the datasets are unavailable — answering ungrounded
    would be worse than declining. Every other failure still yields a valid
    response body, because a malformed response scores zero.
    """
    if not STORE.ready:
        raise HTTPException(
            status_code=503,
            detail=STORE.error or "datasets are still loading",
        )
    started = time.monotonic()
    try:
        result = await answer_question(request.question)
    except Exception:  # noqa: BLE001 - contract requires a valid response
        logger.exception("unhandled failure answering %r", request.question)
        result = {
            "answer": (
                "The agent encountered an internal error and could not produce "
                "a grounded answer for this question."
            ),
            "steps": 0,
            "tool_trace": [],
        }
    elapsed = time.monotonic() - started

    async with _RECENT_LOCK:
        _RECENT.append(
            {
                "question": request.question,
                "answer": result.get("answer", ""),
                "steps": result.get("steps"),
                "tools": [t["tool"] for t in result.get("tool_trace", [])],
                "latency_seconds": round(elapsed, 3),
                "slow": elapsed > 60,
                "diagnostics": result.get("diagnostics", {}),
                "at": time.time(),
            }
        )
        del _RECENT[:-_RECENT_LIMIT]

    if elapsed > 60:
        logger.warning(
            "response took %.1fs (>60s incurs a 20%% point deduction): %r",
            elapsed,
            request.question,
        )
    return result


# ---------------------------------------------------------------------------
# Dashboard support (not part of the graded contract)
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def status() -> dict[str, Any]:
    coverage = STORE.coverage() if STORE.ready else {}
    async with _RECENT_LOCK:
        recent = list(reversed(_RECENT[-20:]))
    latencies = [r["latency_seconds"] for r in recent]
    return {
        "ready": STORE.ready,
        "error": STORE.error,
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
        "config": config.SETTINGS.public_dict(),
        "synthesis_live": config.synthesis_is_live(),
        "datasets": coverage,
        "load_stats": STORE.stats,
        "afr_index_ready": STORE.afr_index.ready,
        "recent": recent,
        "latency": {
            "count": len(latencies),
            "avg": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "max": max(latencies) if latencies else None,
            "slow_responses": sum(1 for r in recent if r["slow"]),
        },
    }


@app.get("/api/public-questions")
async def public_questions() -> dict[str, Any]:
    """The 15 calibration cases, for the dashboard's question picker."""
    if not PUBLIC_QUESTIONS.exists():
        return {"questions": [], "detail": f"not found: {PUBLIC_QUESTIONS}"}
    items = []
    for line in PUBLIC_QUESTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        grading = case.get("grading", {})
        items.append(
            {
                "id": case.get("id"),
                "prompt": case.get("prompt"),
                "difficulty": case.get("difficulty"),
                "datasets": case.get("datasets", []),
                "dataset_scope": case.get("dataset_scope"),
                "reference_answer": case.get("reference_answer"),
                "max_score": grading.get("max_score"),
                "components": [
                    {
                        "component_id": c.get("component_id"),
                        "points": c.get("points"),
                        "expected_fact": c.get("expected_fact"),
                    }
                    for c in grading.get("components", [])
                ],
            }
        )
    return {"questions": items}


_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


class ScoreRequest(BaseModel):
    answer: str
    expected_facts: list[str]


@app.post("/api/score")
async def score(request: ScoreRequest) -> dict[str, Any]:
    """Cheap local proxy for the organizers' LLM judge.

    Checks whether each expected fact's numbers and dates all appear in the
    answer. This is a *screening* heuristic for the dashboard — it is stricter
    than the real judge on wording and blind to synonyms, so treat a miss as
    "look at this", not as a verdict.
    """

    def numbers(text: str) -> set[str]:
        return {
            n.replace(",", "").rstrip(".").lstrip("+")
            for n in _NUMBER_RE.findall(text)
        }

    answer_numbers = numbers(request.answer)
    answer_lower = request.answer.lower()
    results = []
    for fact in request.expected_facts:
        wanted = numbers(fact)
        missing = sorted(wanted - answer_numbers)
        # Sentiment/direction facts carry no numbers; fall back to keywords.
        keywords = [
            w
            for w in re.findall(r"[a-z]{4,}", fact.lower())
            if w not in {"the", "and", "with", "that", "from", "were", "this"}
        ]
        keyword_hits = sum(1 for w in keywords if w in answer_lower)
        results.append(
            {
                "expected_fact": fact,
                "numbers_expected": sorted(wanted),
                "numbers_missing": missing,
                "keyword_coverage": (
                    round(keyword_hits / len(keywords), 2) if keywords else None
                ),
                "likely_pass": not missing and (not keywords or keyword_hits >= max(1, len(keywords) // 3)),
            }
        )
    return {
        "results": results,
        "likely_passed": sum(1 for r in results if r["likely_pass"]),
        "total": len(results),
        "caveat": (
            "Heuristic screening only. The official judge is an LLM that "
            "accepts synonyms and equivalent formatting."
        ),
    }


@app.get("/")
async def dashboard() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="dashboard not built")
    return FileResponse(index)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level=config.LOG_LEVEL.lower(),
    )
