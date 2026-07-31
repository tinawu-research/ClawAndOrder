"""Runtime configuration, read exclusively from environment variables.

Nothing in this module hard-codes an endpoint, credential, hostname or IP.
The organizer-supplied ``~/team.env`` is the source of truth on the cluster:

    source ~/team.env

Every value below has a local-development default so the agent can be started
and exercised without the cluster, but the defaults are deliberately
``localhost``-only and carry no secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    """Read an env var, treating empty/whitespace as unset."""
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------
# Default resolves to the repository's sibling ``data set/`` directory, which is
# where the organizer-supplied corpus lives in the Atom environment. The three
# subdirectory names are the organizer's, spaces included.
_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(_env("DATA_ROOT", str(_REPO_ROOT / "data set")))
RBA_DIR = Path(_env("RBA_DIR", str(DATA_ROOT / "RBA Rates")))
ASX_DIR = Path(_env("ASX_DIR", str(DATA_ROOT / "ASX")))
AFR_DIR = Path(_env("AFR_DIR", str(DATA_ROOT / "AFR")))

# Cap the number of AFR month files loaded. 0 means "all 85". Only for local
# development — leaving this set during evaluation will produce wrong counts.
AFR_MAX_FILES = _env_int("AFR_MAX_FILES", 0)

# Build the AFR token prefilter index at startup. Costs memory and warm-up time
# but turns a full-corpus regex scan into a dict lookup. See datastore.py.
AFR_BUILD_INDEX = _env_bool("AFR_BUILD_INDEX", True)


# ---------------------------------------------------------------------------
# Model routing (LiteLLM proxy)
# ---------------------------------------------------------------------------
# LITELLM_URL is the name used in the organizer handout; LITELLM_BASE_URL is the
# name used in Setup_Instructions.md. Accept either, preferring the handout.
LITELLM_URL = _env(
    "LITELLM_URL", _env("LITELLM_BASE_URL", "http://localhost:4000/v1")
)
LITELLM_KEY = _env("LITELLM_KEY", "sk-local-cluster")

#: Supplied Qwen3.6-35B-A3B-FP8. Owns planning, tool selection and tool-call
#: generation. Never fine-tuned by participants.
BRAIN_MODEL = _env("BRAIN_MODEL", "agent-brain")

#: The team's fine-tuned Nemotron adapter. Owns final answer synthesis only.
DOMAIN_FT_MODEL = _env("DOMAIN_FT_MODEL", "domain-ft")

#: ``mock`` uses the deterministic template synthesiser so the pipeline can be
#: integration-tested before the adapter exists. ``llm`` routes synthesis to
#: DOMAIN_FT_MODEL. MUST be ``llm`` for official evaluation.
DOMAIN_PREDICT_MODE = _env("DOMAIN_PREDICT_MODE", "mock").lower()

EMBED_MODEL = _env("EMBED_MODEL", "")
QDRANT_URL = _env("QDRANT_URL", "")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "afr")


# ---------------------------------------------------------------------------
# Agent loop budget
# ---------------------------------------------------------------------------
#: Hard ceiling on brain<->tool iterations. The scoring rules deduct 20% beyond
#: 60s and zero the question beyond 300s, so the loop is bounded by both a step
#: count and a wall-clock deadline.
MAX_AGENT_STEPS = _env_int("MAX_AGENT_STEPS", 8)

#: Wall-clock budget for the whole /query request, synthesis included. Sits
#: below the 60s mark where the scorer deducts 20% of earned points.
#:
#: 54 rather than 50: with BRAIN_DISABLE_THINKING the median turn dropped from
#: ~18s to ~7s, but the slowest *multi-part* questions (five named tickers, or a
#: count plus a return plus a target) legitimately need 3-4 turns. At 50 the
#: planning budget was 30s, which fit two turns and truncated the third — three
#: public questions ran out of clock mid-gathering. Worst case is still
#: 34s planning + 20s synthesis = 54s, inside the 60s cliff.
QUERY_DEADLINE_SECONDS = _env_int("QUERY_DEADLINE_SECONDS", 54)

#: Slice of QUERY_DEADLINE_SECONDS held back for the final synthesis call.
#: The planning loop must stop this far before the deadline, otherwise the two
#: phases add up: a loop that runs to 50s followed by a 25s synthesis call would
#: answer at 75s and lose 20% of the points it had earned.
#:
#: Keep this at or above SYNTHESIS_TIMEOUT_SECONDS. A reserve smaller than the
#: call it reserves for is not a reserve — the worst case spills back over the
#: request deadline by the difference.
SYNTHESIS_RESERVE_SECONDS = _env_int("SYNTHESIS_RESERVE_SECONDS", 20)

#: Ceiling on one planning attempt.
#:
#: 18, not 22, and the reason is the interaction with the retry ladder rather
#: than the timeout itself. Attempts are front-loaded out of the planning
#: budget, so a 22s first attempt inside a 30s budget left 8s behind it — too
#: short to be a real second chance, yet still spent. One question failed
#: exactly that way: 22s timeout, 8s timeout, zero tool calls, zero points.
#:
#: Measured turn latency with thinking disabled is ~6s median and ~16s at the
#: p95 under the harness's three concurrent questions, so 18s still covers a
#: legitimately slow turn while letting two attempts fit the budget (18 + 16).
BRAIN_TIMEOUT_SECONDS = _env_int("BRAIN_TIMEOUT_SECONDS", 18)
SYNTHESIS_TIMEOUT_SECONDS = _env_int("SYNTHESIS_TIMEOUT_SECONDS", 20)

#: Attempts for one planning turn. The brain is a shared 35B model and the
#: harness sends three questions at once, so an occasional read timeout is
#: expected rather than exceptional — and without a retry a single one costs the
#: whole question, because a planner that never answered gathers no evidence.
#:
#: Two, not three: the planning deadline is split between the allowed attempts,
#: so a third would leave each one too short to finish a real turn.
BRAIN_MAX_ATTEMPTS = _env_int("BRAIN_MAX_ATTEMPTS", 2)

#: Shortest attempt worth starting. Below this the remaining budget is better
#: spent on synthesising the evidence already gathered.
BRAIN_MIN_ATTEMPT_SECONDS = _env_int("BRAIN_MIN_ATTEMPT_SECONDS", 6)

#: Pause before a retry that failed on a server-side status (429/503). Timeouts
#: are not delayed further — the wait already happened.
BRAIN_RETRY_BACKOFF_SECONDS = float(_env("BRAIN_RETRY_BACKOFF_SECONDS", "0.5"))

#: Ceiling on planner output per turn. Unbounded, vLLM will happily generate
#: until the context window is full, and the reasoning chain is what pushes a
#: turn past its timeout. A planning turn only needs a short deliberation plus
#: one tool call, so capping output caps latency.
#:
#: Sized against measured throughput: three concurrent questions share the brain
#: at roughly 42 output tokens/second, so 600 tokens is about 14s — inside one
#: attempt's slice of the planning deadline. Raising this without raising the
#: deadline just converts successful turns into timeouts.
BRAIN_MAX_OUTPUT_TOKENS = _env_int("BRAIN_MAX_OUTPUT_TOKENS", 600)

#: Send ``chat_template_kwargs={"enable_thinking": false}`` on every planning
#: request. Qwen3.6 otherwise opens each turn by narrating its deliberation
#: ("The user is asking three things: 1. ...") before emitting the tool call,
#: and that preamble is the single largest cost in the planning loop.
#:
#: Measured on the brain, same question, 3 concurrent requests:
#:
#:     enable_thinking     latency   output tokens   finish_reason
#:     (default) true      11.6s     600 (capped)    length  <- truncated
#:     false                4.8s     224             stop
#:
#: Truncation is not merely slow, it is a zero: the cap is reached mid-preamble,
#: so no tool call is ever emitted, the turn is retried with less clock, and the
#: question can end with no evidence at all. Three of the fifteen public
#: questions failed exactly that way.
#:
#: The organizers' vLLM is launched with ``--override-generation-config
#: '{"enable_thinking": false}'``, but that governs sampling defaults and does
#: not reach the chat template, so it has to be sent per request.
#:
#: Applies to the planning brain only. Synthesis targets Nemotron, a different
#: model family with its own thinking convention, so it is left alone.
BRAIN_DISABLE_THINKING = _env_bool("BRAIN_DISABLE_THINKING", True)

#: Ceiling on synthesis output. Graded answers are one to three sentences.
SYNTHESIS_MAX_OUTPUT_TOKENS = _env_int("SYNTHESIS_MAX_OUTPUT_TOKENS", 600)

#: The harness sends 3 concurrent questions by default. Tool execution is
#: CPU-bound, so it runs in a bounded thread pool rather than on the event loop.
TOOL_WORKERS = _env_int("TOOL_WORKERS", 4)

SERVER_HOST = _env("SERVER_HOST", "0.0.0.0")

#: Port the evaluation harness connects to.
#:
#: The package does not mandate a port for the agent: the harness reads
#: ``agent.endpoint`` straight out of ``submission.json``, so any port works
#: provided the two agree. For reference, the participant material puts the agent
#: on 5000 (02_execution_guide.md's uvicorn command, submission_template.json)
#: and reserves 8001 for the fine-tuned Nemotron vLLM on the *model* node.
#:
#: We run on 8001 of the head node, which is free there because the adapter is
#: served from the other node. Whoever changes this must change
#: ``submission.json`` to match, and must not co-locate the agent with the
#: adapter's vLLM on one node without moving one of them.
SERVER_PORT = _env_int("SERVER_PORT", 8001)

LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
#: Directory for JSONL request traces. Points at the repo's ``logs/`` folder,
#: which the submission guide requires to contain non-sensitive run logs.
LOG_DIR = Path(_env("LOG_DIR", str(_REPO_ROOT / "logs")))


# ---------------------------------------------------------------------------
# Domain constants that appear verbatim in the question bank
# ---------------------------------------------------------------------------
#: "non-Tabcorp basket" appears in five of the fifteen public questions. It is
#: the 17 tickers remaining after Tabcorp is excluded. Setup_Instructions and
#: the scoring checklist both spell the exclusion as TAH.AX.
TABCORP_TICKER = "TAH.AX"


@dataclass(frozen=True)
class Settings:
    """Snapshot of the effective configuration, for /health and the README."""

    brain_model: str = BRAIN_MODEL
    domain_ft_model: str = DOMAIN_FT_MODEL
    domain_predict_mode: str = DOMAIN_PREDICT_MODE
    litellm_url: str = LITELLM_URL
    max_agent_steps: int = MAX_AGENT_STEPS
    query_deadline_seconds: int = QUERY_DEADLINE_SECONDS
    data_root: str = field(default_factory=lambda: str(DATA_ROOT))

    def public_dict(self) -> dict[str, object]:
        """Configuration safe to expose over HTTP — never includes the key."""
        return {
            "brain_model": self.brain_model,
            "domain_ft_model": self.domain_ft_model,
            "domain_predict_mode": self.domain_predict_mode,
            "litellm_url": self.litellm_url,
            "max_agent_steps": self.max_agent_steps,
            "query_deadline_seconds": self.query_deadline_seconds,
            "data_root": self.data_root,
        }


SETTINGS = Settings()


def plan_deadline_seconds() -> float:
    """Elapsed time at which the planning loop must stop and synthesise.

    Leaves SYNTHESIS_RESERVE_SECONDS of the request budget for the final call,
    so planning and synthesis cannot serialise into a slow-penalty response.
    """
    return max(float(BRAIN_MIN_ATTEMPT_SECONDS), QUERY_DEADLINE_SECONDS - SYNTHESIS_RESERVE_SECONDS)


def synthesis_is_live() -> bool:
    """True when final synthesis actually routes to the fine-tuned model.

    ``DOMAIN_PREDICT_MODE=mock`` is a bootstrap mode for plumbing tests. The
    Challenge Brief requires ``llm`` before official evaluation, so /health
    surfaces this as a warning rather than letting it pass silently.
    """
    return DOMAIN_PREDICT_MODE == "llm"
