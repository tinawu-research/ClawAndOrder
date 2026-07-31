"""Build training prompts using the *production* rendering functions.

Train/serve prompt drift is the classic way a LoRA quietly loses most of its
value: the adapter learns one prefix and meets another at inference. The only
robust defence is to not have two copies of the code, so this module imports
``SYNTHESIS_SYSTEM_PROMPT`` and ``format_evidence`` from the served agent
rather than reimplementing them.

``verify.py`` asserts byte-identity between what this produces and what the
live agent sends, which is the check that keeps the guarantee honest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AGENT_DIR = REPO_ROOT / "src" / "agent"

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from compact import compact_payload  # noqa: E402
from synthesis import (  # noqa: E402
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT_LONG,
    format_evidence,
)

__all__ = [
    "SYNTHESIS_SYSTEM_PROMPT",
    "SYNTHESIS_SYSTEM_PROMPT_LONG",
    "build_messages",
    "make_trace_entry",
    "user_content",
]


def make_trace_entry(tool: str, args: dict[str, Any], payload: Any) -> dict[str, Any]:
    """Build one trace entry in the shape the orchestrator produces.

    ``compact`` is what the model actually reads; ``result`` is the JSON the
    organizers see in ``tool_trace``. Both are carried so a generated example
    and a live one are interchangeable.
    """
    return {
        "tool": tool,
        "args": args,
        "result": json.dumps(payload, default=str, ensure_ascii=False),
        "compact": compact_payload(payload),
        "ok": True,
    }


def user_content(question: str, tool_trace: list[dict[str, Any]]) -> str:
    """The user turn, byte-identical to ``synthesis.synthesize_llm``."""
    return (
        f"Question:\n{question}\n\n"
        f"Verified tool results:\n{format_evidence(tool_trace)}\n\n"
        "Write the final answer."
    )


def build_messages(
    question: str,
    tool_trace: list[dict[str, Any]],
    answer: str | None = None,
    *,
    long_prompt: bool = False,
) -> list[dict[str, str]]:
    """Assemble the full chat turn list for one training example.

    ``long_prompt`` selects the pre-fine-tune system prompt, used only for the
    base-arm control in the comparison.
    """
    system = SYNTHESIS_SYSTEM_PROMPT_LONG if long_prompt else SYNTHESIS_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content(question, tool_trace)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
