import os, json, yaml, time, requests, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# === Cluster configuration ===================================================
# Two roles, two models. We fine-tune Nemotron only — Qwen is supplied as-is.
#
#   brain/agent node  -> Qwen3.6-35B-A3B-FP8, served by vLLM on :8000.
#                        Plans the answer and emits tool calls. Never trained.
#   fine-tune node    -> Llama-3.1-Nemotron-Nano-8B-v1 + LoRA, served on :8001.
#                        Synthesises the final answer from verified tool results.
#
# Endpoints stay in env vars (Setup_Instructions.md) so nothing is pinned to a
# machine; the values below are the local single-box defaults.

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Orchestrator: Qwen (inference only, not a training target) --------------
BRAIN_MODEL      = os.getenv("BRAIN_MODEL", "agent-brain")
BRAIN_MODEL_ID   = "Qwen/Qwen3.6-35B-A3B-Instruct-FP8"   # --served-model-name on :8000
BRAIN_HOST       = os.getenv("BRAIN_HOST", "http://localhost:8000")
BRAIN_CONTAINER  = "vllm-brain"

# --- Fine-tune target: Nemotron ----------------------------------------------
BASE_MODEL       = os.getenv(
    "BASE_MODEL",
    "/home/cognitivo_g17/local-llm-setup/models/Llama-3.1-Nemotron-Nano-8B-v1",
)
DOMAIN_FT_MODEL  = os.getenv("DOMAIN_FT_MODEL", "domain-ft")
DOMAIN_FT_HOST   = os.getenv("DOMAIN_FT_HOST", "http://localhost:8001")

# --- LiteLLM proxy that fronts both aliases ----------------------------------
LITELLM_URL      = os.getenv("LITELLM_URL", "http://localhost:4000/v1")
LITELLM_KEY      = os.getenv("LITELLM_KEY", "sk-local-cluster")

# --- Container images (both already pulled locally) --------------------------
NEMO_IMAGE       = "nvcr.io/nvidia/nemo:25.09"           # handout baseline; 25.04 crashes on GB10
AUTOMODEL_IMAGE  = "nvcr.io/nvidia/nemo-automodel:26.06"

# --- Training hyperparameters (confirmed baseline, 01_training_guide.md) ------
MAX_STEPS        = 100
BATCH_SIZE       = 2
GRAD_ACCUM       = 4        # effective batch = 8
LORA_RANK        = 32
LR               = 5e-5     # 1e-4 spikes the loss at warmup step 50
MAX_SEQ_LEN      = 512      # >512 OOMs on a single GB10
WARMUP_STEPS     = 50
CHECKPOINT_EVERY = 20       # step-20 checkpoint is already worth evaluating

# --- Datasets ----------------------------------------------------------------
AFR_DIR  = REPO_ROOT / "data set" / "AFR"
ASX_DIR  = REPO_ROOT / "data set" / "ASX"
RBA_FILE = REPO_ROOT / "data set" / "RBA Rates" / "RBA-rates.jsonl"

os.environ.update(
    BRAIN_MODEL=BRAIN_MODEL,
    BRAIN_HOST=BRAIN_HOST,
    BASE_MODEL=BASE_MODEL,
    DOMAIN_FT_MODEL=DOMAIN_FT_MODEL,
    DOMAIN_FT_HOST=DOMAIN_FT_HOST,
    LITELLM_URL=LITELLM_URL,
)
os.makedirs("automodel_recipes", exist_ok=True)

print("brain (orchestrator) =", BRAIN_MODEL, "->", BRAIN_MODEL_ID, "@", BRAIN_HOST)
print("base model (to tune) =", BASE_MODEL)
print("fine-tuned alias     =", DOMAIN_FT_MODEL, "@", DOMAIN_FT_HOST)
print("NeMo image           =", NEMO_IMAGE)
print("AutoModel image      =", AUTOMODEL_IMAGE)

